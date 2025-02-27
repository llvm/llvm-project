//===- ConvertToFIR.cpp - Convert HLFIR to FIR ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file defines a pass to lower HLFIR to FIR
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "flang/Optimizer/Builder/Runtime/Assign.h"
#include "flang/Optimizer/Builder/Runtime/Derived.h"
#include "flang/Optimizer/Builder/Runtime/Inquiry.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace hlfir {
#define GEN_PASS_DEF_CONVERTHLFIRTOFIR
#include "flang/Optimizer/HLFIR/Passes.h.inc"
} // namespace hlfir

using namespace mlir;

namespace {
/// May \p lhs alias with \p rhs?
/// TODO: implement HLFIR alias analysis.
class AssignOpConversion : public mlir::OpRewritePattern<hlfir::AssignOp> {
public:
  explicit AssignOpConversion(mlir::MLIRContext *ctx) : OpRewritePattern{ctx} {}

  llvm::LogicalResult
  matchAndRewrite(hlfir::AssignOp assignOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = assignOp->getLoc();
    hlfir::Entity lhs(assignOp.getLhs());
    hlfir::Entity rhs(assignOp.getRhs());
    auto module = assignOp->getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, module);

    if (mlir::isa<hlfir::ExprType>(rhs.getType())) {
      mlir::emitError(loc, "hlfir must be bufferized with --bufferize-hlfir "
                           "pass before being converted to FIR");
      return mlir::failure();
    }
    auto [rhsExv, rhsCleanUp] =
        hlfir::translateToExtendedValue(loc, builder, rhs);
    auto [lhsExv, lhsCleanUp] =
        hlfir::translateToExtendedValue(loc, builder, lhs);
    assert(!lhsCleanUp && !rhsCleanUp &&
           "variable to fir::ExtendedValue must not require cleanup");

    auto emboxRHS = [&](fir::ExtendedValue &rhsExv) -> mlir::Value {
      // There may be overlap between lhs and rhs. The runtime is able to detect
      // and to make a copy of the rhs before modifying the lhs if needed.
      // The code below relies on this and does not do any compile time alias
      // analysis.
      const bool rhsIsValue = fir::isa_trivial(fir::getBase(rhsExv).getType());
      if (rhsIsValue) {
        // createBox can only be called for fir::ExtendedValue that are
        // already in memory. Place the integer/real/complex/logical scalar
        // in memory.
        // The RHS might be i1, which is not supported for emboxing.
        // If LHS is not polymorphic, we may cast the RHS to the LHS type
        // before emboxing. If LHS is polymorphic we have to figure out
        // the data type for RHS emboxing anyway.
        // It is probably a good idea to make sure that the data type
        // of the RHS is always a valid Fortran storage data type.
        // For the time being, just handle i1 explicitly here.
        mlir::Type rhsType = rhs.getFortranElementType();
        mlir::Value rhsVal = fir::getBase(rhsExv);
        if (rhsType == builder.getI1Type()) {
          rhsType = fir::LogicalType::get(builder.getContext(), 4);
          rhsVal = builder.createConvert(loc, rhsType, rhsVal);
        }
        mlir::Value temp = builder.create<fir::AllocaOp>(loc, rhsType);
        builder.create<fir::StoreOp>(loc, rhsVal, temp);
        rhsExv = temp;
      }
      return fir::getBase(builder.createBox(loc, rhsExv));
    };

    if (assignOp.isAllocatableAssignment()) {
      // Whole allocatable assignment: use the runtime to deal with the
      // reallocation.
      mlir::Value from = emboxRHS(rhsExv);
      mlir::Value to = fir::getBase(lhsExv);
      if (assignOp.mustKeepLhsLengthInAllocatableAssignment()) {
        // Indicate the runtime that it should not reallocate in case of length
        // mismatch, and that it should use the LHS explicit/assumed length if
        // allocating/reallocation the LHS.
        // Note that AssignExplicitLengthCharacter() must be used
        // when isTemporaryLHS() is true here: the LHS is known to be
        // character allocatable in this case, so finalization will not
        // happen (as implied by temporary_lhs attribute), and LHS
        // must keep its length (as implied by keep_lhs_length_if_realloc).
        fir::runtime::genAssignExplicitLengthCharacter(builder, loc, to, from);
      } else if (assignOp.isTemporaryLHS()) {
        // Use AssignTemporary, when the LHS is a compiler generated temporary.
        // Note that it also works properly for polymorphic LHS (i.e. the LHS
        // will have the RHS dynamic type after the assignment).
        fir::runtime::genAssignTemporary(builder, loc, to, from);
      } else if (lhs.isPolymorphic()) {
        // Indicate the runtime that the LHS must have the RHS dynamic type
        // after the assignment.
        fir::runtime::genAssignPolymorphic(builder, loc, to, from);
      } else {
        fir::runtime::genAssign(builder, loc, to, from);
      }
    } else if (lhs.isArray() ||
               // Special case for element-by-element (or scalar) assignments
               // generated for creating polymorphic expressions.
               // The LHS of these assignments is a box describing just
               // a single element, not the whole allocatable temp.
               // They do not have 'realloc' attribute, because reallocation
               // must not happen. The only expected effect of such an
               // assignment is the copy of the contents, because the dynamic
               // types of the LHS and the RHS must match already. We use the
               // runtime in this case so that the polymorphic (including
               // unlimited) content is copied properly.
               (lhs.isPolymorphic() && assignOp.isTemporaryLHS())) {
      // Use the runtime for simplicity. An optimization pass will be added to
      // inline array assignment when profitable.
      mlir::Value from = emboxRHS(rhsExv);
      mlir::Value to = fir::getBase(builder.createBox(loc, lhsExv));
      // This is not a whole allocatable assignment: the runtime will not
      // reallocate and modify "toMutableBox" even if it is taking it by
      // reference.
      auto toMutableBox = builder.createTemporary(loc, to.getType());
      builder.create<fir::StoreOp>(loc, to, toMutableBox);
      if (assignOp.isTemporaryLHS())
        fir::runtime::genAssignTemporary(builder, loc, toMutableBox, from);
      else
        fir::runtime::genAssign(builder, loc, toMutableBox, from);
    } else {
      // TODO: use the type specification to see if IsFinalizable is set,
      // or propagate IsFinalizable attribute from lowering.
      bool needFinalization =
          !assignOp.isTemporaryLHS() &&
          mlir::isa<fir::RecordType>(fir::getElementTypeOf(lhsExv));

      // genScalarAssignment() must take care of potential overlap
      // between LHS and RHS. Note that the overlap is possible
      // also for components of LHS/RHS, and the Assign() runtime
      // must take care of it.
      fir::factory::genScalarAssignment(builder, loc, lhsExv, rhsExv,
                                        needFinalization,
                                        assignOp.isTemporaryLHS());
    }
    rewriter.eraseOp(assignOp);
    return mlir::success();
  }
};

class CopyInOpConversion : public mlir::OpRewritePattern<hlfir::CopyInOp> {
public:
  explicit CopyInOpConversion(mlir::MLIRContext *ctx) : OpRewritePattern{ctx} {}

  struct CopyInResult {
    mlir::Value addr;
    mlir::Value wasCopied;
  };

  static CopyInResult genNonOptionalCopyIn(mlir::Location loc,
                                           fir::FirOpBuilder &builder,
                                           hlfir::CopyInOp copyInOp) {
    mlir::Value inputVariable = copyInOp.getVar();
    mlir::Type resultAddrType = copyInOp.getCopiedIn().getType();
    mlir::Value isContiguous =
        fir::runtime::genIsContiguous(builder, loc, inputVariable);
    mlir::Value addr =
        builder
            .genIfOp(loc, {resultAddrType}, isContiguous,
                     /*withElseRegion=*/true)
            .genThen(
                [&]() { builder.create<fir::ResultOp>(loc, inputVariable); })
            .genElse([&] {
              // Create temporary on the heap. Note that the runtime is used and
              // that is desired: since the data copy happens under a runtime
              // check (for IsContiguous) the copy loops can hardly provide any
              // value to optimizations, instead, the optimizer just wastes
              // compilation time on these loops.
              mlir::Value temp = copyInOp.getTempBox();
              fir::runtime::genCopyInAssign(builder, loc, temp, inputVariable);
              mlir::Value copy = builder.create<fir::LoadOp>(loc, temp);
              // Get rid of allocatable flag in the fir.box.
              if (mlir::cast<fir::BaseBoxType>(resultAddrType).isAssumedRank())
                copy = builder.create<fir::ReboxAssumedRankOp>(
                    loc, resultAddrType, copy,
                    fir::LowerBoundModifierAttribute::Preserve);
              else
                copy = builder.create<fir::ReboxOp>(loc, resultAddrType, copy,
                                                    /*shape=*/mlir::Value{},
                                                    /*slice=*/mlir::Value{});
              builder.create<fir::ResultOp>(loc, copy);
            })
            .getResults()[0];
    return {addr, builder.genNot(loc, isContiguous)};
  }

  static CopyInResult genOptionalCopyIn(mlir::Location loc,
                                        fir::FirOpBuilder &builder,
                                        hlfir::CopyInOp copyInOp) {
    mlir::Type resultAddrType = copyInOp.getCopiedIn().getType();
    mlir::Value isPresent = copyInOp.getVarIsPresent();
    auto res =
        builder
            .genIfOp(loc, {resultAddrType, builder.getI1Type()}, isPresent,
                     /*withElseRegion=*/true)
            .genThen([&]() {
              CopyInResult res = genNonOptionalCopyIn(loc, builder, copyInOp);
              builder.create<fir::ResultOp>(
                  loc, mlir::ValueRange{res.addr, res.wasCopied});
            })
            .genElse([&] {
              mlir::Value absent =
                  builder.create<fir::AbsentOp>(loc, resultAddrType);
              builder.create<fir::ResultOp>(
                  loc, mlir::ValueRange{absent, isPresent});
            })
            .getResults();
    return {res[0], res[1]};
  }

  llvm::LogicalResult
  matchAndRewrite(hlfir::CopyInOp copyInOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = copyInOp.getLoc();
    fir::FirOpBuilder builder(rewriter, copyInOp.getOperation());
    CopyInResult result = copyInOp.getVarIsPresent()
                              ? genOptionalCopyIn(loc, builder, copyInOp)
                              : genNonOptionalCopyIn(loc, builder, copyInOp);
    rewriter.replaceOp(copyInOp, {result.addr, result.wasCopied});
    return mlir::success();
  }
};

class CopyOutOpConversion : public mlir::OpRewritePattern<hlfir::CopyOutOp> {
public:
  explicit CopyOutOpConversion(mlir::MLIRContext *ctx)
      : OpRewritePattern{ctx} {}

  llvm::LogicalResult
  matchAndRewrite(hlfir::CopyOutOp copyOutOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = copyOutOp.getLoc();
    fir::FirOpBuilder builder(rewriter, copyOutOp.getOperation());

    builder.genIfThen(loc, copyOutOp.getWasCopied())
        .genThen([&]() {
          mlir::Value temp = copyOutOp.getTemp();
          mlir::Value varMutableBox;
          // Generate CopyOutAssign runtime call.
          if (mlir::Value var = copyOutOp.getVar()) {
            // Set the variable descriptor pointer in order to copy data from
            // the temporary to the actualArg. Note that in case the actual
            // argument is ALLOCATABLE/POINTER the CopyOutAssign()
            // implementation should not engage its reallocation, because the
            // temporary is rank, shape and type compatible with it. Moreover,
            // CopyOutAssign() guarantees that there will be no finalization for
            // the LHS even if it is of a derived type with finalization.
            varMutableBox = builder.createTemporary(loc, var.getType());
            builder.create<fir::StoreOp>(loc, var, varMutableBox);
          } else {
            // Even when there is no need to copy back the data (e.g., the dummy
            // argument was intent(in), CopyOutAssign is called to
            // destroy/deallocate the temporary.
            varMutableBox = builder.create<fir::ZeroOp>(loc, temp.getType());
          }
          fir::runtime::genCopyOutAssign(builder, loc, varMutableBox,
                                         copyOutOp.getTemp());
        })
        .end();
    rewriter.eraseOp(copyOutOp);
    return mlir::success();
  }
};

class DeclareOpConversion : public mlir::OpRewritePattern<hlfir::DeclareOp> {
public:
  explicit DeclareOpConversion(mlir::MLIRContext *ctx)
      : OpRewritePattern{ctx} {}

  llvm::LogicalResult
  matchAndRewrite(hlfir::DeclareOp declareOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = declareOp->getLoc();
    mlir::Value memref = declareOp.getMemref();
    fir::FortranVariableFlagsAttr fortranAttrs;
    cuf::DataAttributeAttr dataAttr;
    if (auto attrs = declareOp.getFortranAttrs())
      fortranAttrs =
          fir::FortranVariableFlagsAttr::get(rewriter.getContext(), *attrs);
    if (auto attr = declareOp.getDataAttr())
      dataAttr = cuf::DataAttributeAttr::get(rewriter.getContext(), *attr);
    auto firDeclareOp = rewriter.create<fir::DeclareOp>(
        loc, memref.getType(), memref, declareOp.getShape(),
        declareOp.getTypeparams(), declareOp.getDummyScope(),
        declareOp.getUniqName(), fortranAttrs, dataAttr);

    // Propagate other attributes from hlfir.declare to fir.declare.
    // OpenACC's acc.declare is one example. Right now, the propagation
    // is verbatim.
    mlir::NamedAttrList elidedAttrs =
        mlir::NamedAttrList{firDeclareOp->getAttrs()};
    for (const mlir::NamedAttribute &attr : declareOp->getAttrs())
      if (!elidedAttrs.get(attr.getName()))
        firDeclareOp->setAttr(attr.getName(), attr.getValue());

    auto firBase = firDeclareOp.getResult();
    mlir::Value hlfirBase;
    mlir::Type hlfirBaseType = declareOp.getBase().getType();
    if (mlir::isa<fir::BaseBoxType>(hlfirBaseType)) {
      fir::FirOpBuilder builder(rewriter, declareOp.getOperation());
      // Helper to generate the hlfir fir.box with the local lower bounds and
      // type parameters.
      auto genHlfirBox = [&]() -> mlir::Value {
        if (auto baseBoxType =
                mlir::dyn_cast<fir::BaseBoxType>(firBase.getType())) {
          // Rebox so that lower bounds are correct.
          if (baseBoxType.isAssumedRank())
            return builder.create<fir::ReboxAssumedRankOp>(
                loc, hlfirBaseType, firBase,
                fir::LowerBoundModifierAttribute::SetToOnes);
          return builder.create<fir::ReboxOp>(loc, hlfirBaseType, firBase,
                                              declareOp.getShape(),
                                              /*slice=*/mlir::Value{});
        } else {
          llvm::SmallVector<mlir::Value> typeParams;
          auto maybeCharType = mlir::dyn_cast<fir::CharacterType>(
              fir::unwrapSequenceType(fir::unwrapPassByRefType(hlfirBaseType)));
          if (!maybeCharType || maybeCharType.hasDynamicLen())
            typeParams.append(declareOp.getTypeparams().begin(),
                              declareOp.getTypeparams().end());
          return builder.create<fir::EmboxOp>(
              loc, hlfirBaseType, firBase, declareOp.getShape(),
              /*slice=*/mlir::Value{}, typeParams);
        }
      };
      if (!mlir::cast<fir::FortranVariableOpInterface>(declareOp.getOperation())
               .isOptional()) {
        hlfirBase = genHlfirBox();
        // If the original base is a box too, we could as well
        // use the HLFIR box as the FIR base: otherwise, the two
        // boxes are "alive" at the same time, and the FIR box
        // is used for accessing the base_addr and the HLFIR box
        // is used for accessing the bounds etc. Using the HLFIR box,
        // that holds the same base_addr at this point, makes
        // the representation a little bit more clear.
        if (hlfirBase.getType() == firBase.getType())
          firBase = hlfirBase;
      } else {
        // Need to conditionally rebox/embox the optional: the input fir.box
        // may be null and the rebox would be illegal. It is also important to
        // preserve the optional aspect: the hlfir fir.box should be null if
        // the entity is absent so that later fir.is_present on the hlfir base
        // are valid.
        mlir::Value isPresent =
            builder.create<fir::IsPresentOp>(loc, builder.getI1Type(), firBase);
        hlfirBase = builder
                        .genIfOp(loc, {hlfirBaseType}, isPresent,
                                 /*withElseRegion=*/true)
                        .genThen([&] {
                          builder.create<fir::ResultOp>(loc, genHlfirBox());
                        })
                        .genElse([&]() {
                          mlir::Value absent =
                              builder.create<fir::AbsentOp>(loc, hlfirBaseType);
                          builder.create<fir::ResultOp>(loc, absent);
                        })
                        .getResults()[0];
      }
    } else if (mlir::isa<fir::BoxCharType>(hlfirBaseType)) {
      assert(declareOp.getTypeparams().size() == 1 &&
             "must contain character length");
      hlfirBase = rewriter.create<fir::EmboxCharOp>(
          loc, hlfirBaseType, firBase, declareOp.getTypeparams()[0]);
    } else {
      if (hlfirBaseType != firBase.getType()) {
        declareOp.emitOpError()
            << "unhandled HLFIR variable type '" << hlfirBaseType << "'\n";
        return mlir::failure();
      }
      hlfirBase = firBase;
    }
    rewriter.replaceOp(declareOp, {hlfirBase, firBase});
    return mlir::success();
  }
};

class DesignateOpConversion
    : public mlir::OpRewritePattern<hlfir::DesignateOp> {
  // Helper method to generate the coordinate of the first element
  // of an array section. It is also called for cases of non-section
  // array element addressing.
  static mlir::Value genSubscriptBeginAddr(
      fir::FirOpBuilder &builder, mlir::Location loc,
      hlfir::DesignateOp designate, mlir::Type baseEleTy, mlir::Value base,
      mlir::Value shape,
      const llvm::SmallVector<mlir::Value> &firBaseTypeParameters) {
    assert(!designate.getIndices().empty());
    llvm::SmallVector<mlir::Value> firstElementIndices;
    auto indices = designate.getIndices();
    int i = 0;
    auto attrs = designate.getIsTripletAttr();
    for (auto isTriplet : attrs.asArrayRef()) {
      // Coordinate of the first element are the index and triplets lower
      // bounds
      firstElementIndices.push_back(indices[i]);
      i = i + (isTriplet ? 3 : 1);
    }
    mlir::Type arrayCoorType = fir::ReferenceType::get(baseEleTy);
    base = builder.create<fir::ArrayCoorOp>(
        loc, arrayCoorType, base, shape,
        /*slice=*/mlir::Value{}, firstElementIndices, firBaseTypeParameters);
    return base;
  }

public:
  explicit DesignateOpConversion(mlir::MLIRContext *ctx)
      : OpRewritePattern{ctx} {}

  llvm::LogicalResult
  matchAndRewrite(hlfir::DesignateOp designate,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = designate.getLoc();
    fir::FirOpBuilder builder(rewriter, designate.getOperation());

    hlfir::Entity baseEntity(designate.getMemref());

    if (baseEntity.isMutableBox())
      TODO(loc, "hlfir::designate load of pointer or allocatable");

    mlir::Type designateResultType = designate.getResult().getType();
    llvm::SmallVector<mlir::Value> firBaseTypeParameters;
    auto [base, shape] = hlfir::genVariableFirBaseShapeAndParams(
        loc, builder, baseEntity, firBaseTypeParameters);
    mlir::Type baseEleTy = hlfir::getFortranElementType(base.getType());
    mlir::Type resultEleTy = hlfir::getFortranElementType(designateResultType);

    mlir::Value fieldIndex;
    if (designate.getComponent()) {
      mlir::Type baseRecordType = baseEntity.getFortranElementType();
      if (fir::isRecordWithTypeParameters(baseRecordType))
        TODO(loc, "hlfir.designate with a parametrized derived type base");
      fieldIndex = builder.create<fir::FieldIndexOp>(
          loc, fir::FieldType::get(builder.getContext()),
          designate.getComponent().value(), baseRecordType,
          /*typeParams=*/mlir::ValueRange{});
      if (baseEntity.isScalar()) {
        // Component refs of scalar base right away:
        // - scalar%scalar_component [substring|complex_part] or
        // - scalar%static_size_array_comp
        // - scalar%array(indices) [substring| complex part]
        mlir::Type componentType =
            mlir::cast<fir::RecordType>(baseEleTy).getType(
                designate.getComponent().value());
        mlir::Type coorTy = fir::ReferenceType::get(componentType);
        base = builder.create<fir::CoordinateOp>(loc, coorTy, base, fieldIndex);
        if (mlir::isa<fir::BaseBoxType>(componentType)) {
          auto variableInterface = mlir::cast<fir::FortranVariableOpInterface>(
              designate.getOperation());
          if (variableInterface.isAllocatable() ||
              variableInterface.isPointer()) {
            rewriter.replaceOp(designate, base);
            return mlir::success();
          }
          TODO(loc,
               "addressing parametrized derived type automatic components");
        }
        baseEleTy = hlfir::getFortranElementType(componentType);
        shape = designate.getComponentShape();
      } else {
        // array%component[(indices) substring|complex part] cases.
        // Component ref of array bases are dealt with below in embox/rebox.
        assert(mlir::isa<fir::BaseBoxType>(designateResultType));
      }
    }

    if (mlir::isa<fir::BaseBoxType>(designateResultType)) {
      // Generate embox or rebox.
      mlir::Type eleTy = fir::unwrapPassByRefType(designateResultType);
      bool isScalarDesignator = !mlir::isa<fir::SequenceType>(eleTy);
      mlir::Value sourceBox;
      if (isScalarDesignator) {
        // The base box will be used for emboxing the scalar element.
        sourceBox = base;
        // Generate the coordinate of the element.
        base = genSubscriptBeginAddr(builder, loc, designate, baseEleTy, base,
                                     shape, firBaseTypeParameters);
        shape = nullptr;
        // Type information will be taken from the source box,
        // so the type parameters are not needed.
        firBaseTypeParameters.clear();
      }
      llvm::SmallVector<mlir::Value> triples;
      llvm::SmallVector<mlir::Value> sliceFields;
      mlir::Type idxTy = builder.getIndexType();
      auto subscripts = designate.getIndices();
      if (fieldIndex && baseEntity.isArray()) {
        // array%scalar_comp or array%array_comp(indices)
        // Generate triples for array(:, :, ...).
        triples = genFullSliceTriples(builder, loc, baseEntity);
        sliceFields.push_back(fieldIndex);
        // Add indices in the field path for "array%array_comp(indices)"
        // case. The indices of components provided to the sliceOp must
        // be zero based (fir.slice has no knowledge of the component
        // lower bounds). The component lower bounds are applied here.
        if (!subscripts.empty()) {
          llvm::SmallVector<mlir::Value> lbounds = hlfir::genLowerbounds(
              loc, builder, designate.getComponentShape(), subscripts.size());
          for (auto [i, lb] : llvm::zip(subscripts, lbounds)) {
            mlir::Value iIdx = builder.createConvert(loc, idxTy, i);
            mlir::Value lbIdx = builder.createConvert(loc, idxTy, lb);
            sliceFields.emplace_back(
                builder.create<mlir::arith::SubIOp>(loc, iIdx, lbIdx));
          }
        }
      } else if (!isScalarDesignator) {
        // Otherwise, this is an array section with triplets.
        auto undef = builder.create<fir::UndefOp>(loc, idxTy);
        unsigned i = 0;
        for (auto isTriplet : designate.getIsTriplet()) {
          triples.push_back(subscripts[i++]);
          if (isTriplet) {
            triples.push_back(subscripts[i++]);
            triples.push_back(subscripts[i++]);
          } else {
            triples.push_back(undef);
            triples.push_back(undef);
          }
        }
      }
      llvm::SmallVector<mlir::Value, 2> substring;
      if (!designate.getSubstring().empty()) {
        substring.push_back(designate.getSubstring()[0]);
        mlir::Type idxTy = builder.getIndexType();
        // fir.slice op substring expects the zero based lower bound.
        mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
        substring[0] = builder.createConvert(loc, idxTy, substring[0]);
        substring[0] =
            builder.create<mlir::arith::SubIOp>(loc, substring[0], one);
        substring.push_back(designate.getTypeparams()[0]);
      }
      if (designate.getComplexPart()) {
        if (triples.empty())
          triples = genFullSliceTriples(builder, loc, baseEntity);
        sliceFields.push_back(builder.createIntegerConstant(
            loc, idxTy, *designate.getComplexPart()));
      }
      mlir::Value slice;
      if (!triples.empty())
        slice =
            builder.create<fir::SliceOp>(loc, triples, sliceFields, substring);
      else
        assert(sliceFields.empty() && substring.empty());
      llvm::SmallVector<mlir::Type> resultType{designateResultType};
      mlir::Value resultBox;
      if (mlir::isa<fir::BaseBoxType>(base.getType()))
        resultBox =
            builder.create<fir::ReboxOp>(loc, resultType, base, shape, slice);
      else
        resultBox =
            builder.create<fir::EmboxOp>(loc, resultType, base, shape, slice,
                                         firBaseTypeParameters, sourceBox);
      rewriter.replaceOp(designate, resultBox);
      return mlir::success();
    }

    // Otherwise, the result is the address of a scalar, or the address of the
    // first element of a contiguous array section with compile time constant
    // shape. The base may be an array, or a scalar.
    mlir::Type resultAddressType = designateResultType;
    if (auto boxCharType =
            mlir::dyn_cast<fir::BoxCharType>(designateResultType))
      resultAddressType = fir::ReferenceType::get(boxCharType.getEleTy());

    // Array element indexing.
    if (!designate.getIndices().empty()) {
      // - array(indices) [substring|complex_part] or
      // - scalar%array_comp(indices) [substring|complex_part]
      // This may be a ranked contiguous array section in which case
      // The first element address is being computed.
      base = genSubscriptBeginAddr(builder, loc, designate, baseEleTy, base,
                                   shape, firBaseTypeParameters);
    }

    // Scalar substring (potentially on the previously built array element or
    // component reference).
    if (!designate.getSubstring().empty())
      base = fir::factory::CharacterExprHelper{builder, loc}.genSubstringBase(
          base, designate.getSubstring()[0], resultAddressType);

    // Scalar complex part ref
    if (designate.getComplexPart()) {
      // Sequence types should have already been handled by this point
      assert(!mlir::isa<fir::SequenceType>(designateResultType));
      auto index = builder.createIntegerConstant(loc, builder.getIndexType(),
                                                 *designate.getComplexPart());
      auto coorTy = fir::ReferenceType::get(resultEleTy);
      base = builder.create<fir::CoordinateOp>(loc, coorTy, base, index);
    }

    // Cast/embox the computed scalar address if needed.
    if (mlir::isa<fir::BoxCharType>(designateResultType)) {
      assert(designate.getTypeparams().size() == 1 &&
             "must have character length");
      auto emboxChar = builder.create<fir::EmboxCharOp>(
          loc, designateResultType, base, designate.getTypeparams()[0]);
      rewriter.replaceOp(designate, emboxChar.getResult());
    } else {
      base = builder.createConvert(loc, designateResultType, base);
      rewriter.replaceOp(designate, base);
    }
    return mlir::success();
  }

private:
  // Generates triple for full slice
  // Used for component and complex part slices when a triple is
  // not specified
  static llvm::SmallVector<mlir::Value>
  genFullSliceTriples(fir::FirOpBuilder &builder, mlir::Location loc,
                      hlfir::Entity baseEntity) {
    llvm::SmallVector<mlir::Value> triples;
    mlir::Type idxTy = builder.getIndexType();
    auto one = builder.createIntegerConstant(loc, idxTy, 1);
    for (auto [lb, ub] : hlfir::genBounds(loc, builder, baseEntity)) {
      triples.push_back(builder.createConvert(loc, idxTy, lb));
      triples.push_back(builder.createConvert(loc, idxTy, ub));
      triples.push_back(one);
    }
    return triples;
  }
};

class ParentComponentOpConversion
    : public mlir::OpRewritePattern<hlfir::ParentComponentOp> {
public:
  explicit ParentComponentOpConversion(mlir::MLIRContext *ctx)
      : OpRewritePattern{ctx} {}

  llvm::LogicalResult
  matchAndRewrite(hlfir::ParentComponentOp parentComponent,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = parentComponent.getLoc();
    mlir::Type resultType = parentComponent.getType();
    if (!mlir::isa<fir::BoxType>(parentComponent.getType())) {
      mlir::Value baseAddr = parentComponent.getMemref();
      // Scalar parent component ref without any length type parameters. The
      // input may be a fir.class if it is polymorphic, since this is a scalar
      // and the output will be monomorphic, the base address can be extracted
      // from the fir.class.
      if (mlir::isa<fir::BaseBoxType>(baseAddr.getType()))
        baseAddr = rewriter.create<fir::BoxAddrOp>(loc, baseAddr);
      rewriter.replaceOpWithNewOp<fir::ConvertOp>(parentComponent, resultType,
                                                  baseAddr);
      return mlir::success();
    }
    // Array parent component ref or PDTs.
    hlfir::Entity base{parentComponent.getMemref()};
    mlir::Value baseAddr = base.getBase();
    if (!mlir::isa<fir::BaseBoxType>(baseAddr.getType())) {
      // Embox cannot directly be used to address parent components: it expects
      // the output type to match the input type when there are no slices. When
      // the types have at least one component, a slice to the first element can
      // be built, and the result set to the parent component type. Just create
      // a fir.box with the base for now since this covers all cases.
      mlir::Type baseBoxType =
          fir::BoxType::get(base.getElementOrSequenceType());
      assert(!base.hasLengthParameters() &&
             "base must be a box if it has any type parameters");
      baseAddr = rewriter.create<fir::EmboxOp>(
          loc, baseBoxType, baseAddr, parentComponent.getShape(),
          /*slice=*/mlir::Value{}, /*typeParams=*/mlir::ValueRange{});
    }
    rewriter.replaceOpWithNewOp<fir::ReboxOp>(parentComponent, resultType,
                                              baseAddr,
                                              /*shape=*/mlir::Value{},
                                              /*slice=*/mlir::Value{});
    return mlir::success();
  }
};

class NoReassocOpConversion
    : public mlir::OpRewritePattern<hlfir::NoReassocOp> {
public:
  explicit NoReassocOpConversion(mlir::MLIRContext *ctx)
      : OpRewritePattern{ctx} {}

  llvm::LogicalResult
  matchAndRewrite(hlfir::NoReassocOp noreassoc,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<fir::NoReassocOp>(noreassoc,
                                                  noreassoc.getVal());
    return mlir::success();
  }
};

class NullOpConversion : public mlir::OpRewritePattern<hlfir::NullOp> {
public:
  explicit NullOpConversion(mlir::MLIRContext *ctx) : OpRewritePattern{ctx} {}

  llvm::LogicalResult
  matchAndRewrite(hlfir::NullOp nullop,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<fir::ZeroOp>(nullop, nullop.getType());
    return mlir::success();
  }
};

class GetExtentOpConversion
    : public mlir::OpRewritePattern<hlfir::GetExtentOp> {
public:
  using mlir::OpRewritePattern<hlfir::GetExtentOp>::OpRewritePattern;

  llvm::LogicalResult
  matchAndRewrite(hlfir::GetExtentOp getExtentOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value shape = getExtentOp.getShape();
    mlir::Operation *shapeOp = shape.getDefiningOp();
    // the hlfir.shape_of operation which led to the creation of this get_extent
    // operation should now have been lowered to a fir.shape operation
    if (auto s = mlir::dyn_cast_or_null<fir::ShapeOp>(shapeOp)) {
      fir::ShapeType shapeTy = mlir::cast<fir::ShapeType>(shape.getType());
      llvm::APInt dim = getExtentOp.getDim();
      uint64_t dimVal = dim.getLimitedValue(shapeTy.getRank());
      mlir::Value extent = s.getExtents()[dimVal];
      fir::FirOpBuilder builder(rewriter, getExtentOp.getOperation());
      extent = builder.createConvert(getExtentOp.getLoc(),
                                     builder.getIndexType(), extent);
      rewriter.replaceOp(getExtentOp, extent);
      return mlir::success();
    }
    return mlir::failure();
  }
};

class ConvertHLFIRtoFIR
    : public hlfir::impl::ConvertHLFIRtoFIRBase<ConvertHLFIRtoFIR> {
public:
  void runOnOperation() override {
    // TODO: like "bufferize-hlfir" pass, runtime signature may be added
    // by this pass. This requires the pass to run on the ModuleOp. It would
    // probably be more optimal to have it run on FuncOp and find a way to
    // generate the signatures in a thread safe way.
    auto module = this->getOperation();
    auto *context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.insert<AssignOpConversion, CopyInOpConversion, CopyOutOpConversion,
                    DeclareOpConversion, DesignateOpConversion,
                    GetExtentOpConversion, NoReassocOpConversion,
                    NullOpConversion, ParentComponentOpConversion>(context);
    mlir::ConversionTarget target(*context);
    target.addIllegalDialect<hlfir::hlfirDialect>();
    target.markUnknownOpDynamicallyLegal(
        [](mlir::Operation *) { return true; });
    if (mlir::failed(mlir::applyPartialConversion(module, target,
                                                  std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(context),
                      "failure in HLFIR to FIR conversion pass");
      signalPassFailure();
    }
  }
};

} // namespace
