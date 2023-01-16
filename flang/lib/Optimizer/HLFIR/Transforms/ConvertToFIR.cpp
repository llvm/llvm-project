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
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "mlir/Transforms/DialectConversion.h"

namespace hlfir {
#define GEN_PASS_DEF_CONVERTHLFIRTOFIR
#include "flang/Optimizer/HLFIR/Passes.h.inc"
} // namespace hlfir

using namespace mlir;

namespace {
/// May \p lhs alias with \p rhs?
/// TODO: implement HLFIR alias analysis.
static bool mayAlias(hlfir::Entity lhs, hlfir::Entity rhs) { return true; }

class AssignOpConversion : public mlir::OpRewritePattern<hlfir::AssignOp> {
public:
  explicit AssignOpConversion(mlir::MLIRContext *ctx) : OpRewritePattern{ctx} {}

  mlir::LogicalResult
  matchAndRewrite(hlfir::AssignOp assignOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = assignOp->getLoc();
    hlfir::Entity lhs(assignOp.getLhs());
    hlfir::Entity rhs(assignOp.getRhs());
    auto module = assignOp->getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, fir::getKindMapping(module));

    if (rhs.getType().isa<hlfir::ExprType>()) {
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

    if (lhs.isArray()) {
      // Use the runtime for simplicity. An optimization pass will be added to
      // inline array assignment when profitable.
      auto to = fir::getBase(builder.createBox(loc, lhsExv));
      auto from = fir::getBase(builder.createBox(loc, rhsExv));
      bool cleanUpTemp = false;
      mlir::Type fromHeapType = fir::HeapType::get(
          fir::unwrapRefType(from.getType().cast<fir::BoxType>().getEleTy()));
      if (mayAlias(rhs, lhs)) {
        /// Use the runtime to make a quick and dirty temp with the rhs value.
        /// Overkill for scalar rhs that could be done in much more clever ways.
        /// Note that temp descriptor must have the allocatable flag set so that
        /// the runtime will allocate it with the shape and type parameters of
        //  the RHS.
        mlir::Type fromBoxHeapType = fir::BoxType::get(fromHeapType);
        auto fromMutableBox = builder.createTemporary(loc, fromBoxHeapType);
        mlir::Value unallocatedBox = fir::factory::createUnallocatedBox(
            builder, loc, fromBoxHeapType, {});
        builder.create<fir::StoreOp>(loc, unallocatedBox, fromMutableBox);
        fir::runtime::genAssign(builder, loc, fromMutableBox, from);
        cleanUpTemp = true;
        from = builder.create<fir::LoadOp>(loc, fromMutableBox);
      }
      auto toMutableBox = builder.createTemporary(loc, to.getType());
      // As per 10.2.1.2 point 1 (1) polymorphic variables must be allocatable.
      // It is assumed here that they have been reallocated with the dynamic
      // type and that the mutableBox will not be modified.
      builder.create<fir::StoreOp>(loc, to, toMutableBox);
      fir::runtime::genAssign(builder, loc, toMutableBox, from);
      if (cleanUpTemp) {
        mlir::Value addr =
            builder.create<fir::BoxAddrOp>(loc, fromHeapType, from);
        builder.create<fir::FreeMemOp>(loc, addr);
      }
    } else {
      // Assume overlap does not matter for scalar (dealt with memmove for
      // characters).
      // This is not true if this is a derived type with "recursive" allocatable
      // components, in which case an overlap would matter because the LHS
      // reallocation, if any, may modify the RHS component value before it is
      // copied into the LHS.
      if (fir::isRecordWithAllocatableMember(lhs.getFortranElementType()))
        TODO(loc, "assignment with allocatable components");
      fir::factory::genScalarAssignment(builder, loc, lhsExv, rhsExv);
    }
    rewriter.eraseOp(assignOp);
    return mlir::success();
  }
};

class DeclareOpConversion : public mlir::OpRewritePattern<hlfir::DeclareOp> {
public:
  explicit DeclareOpConversion(mlir::MLIRContext *ctx)
      : OpRewritePattern{ctx} {}

  mlir::LogicalResult
  matchAndRewrite(hlfir::DeclareOp declareOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = declareOp->getLoc();
    mlir::Value memref = declareOp.getMemref();
    fir::FortranVariableFlagsAttr fortranAttrs;
    if (auto attrs = declareOp.getFortranAttrs())
      fortranAttrs =
          fir::FortranVariableFlagsAttr::get(rewriter.getContext(), *attrs);
    auto firBase = rewriter
                       .create<fir::DeclareOp>(
                           loc, memref.getType(), memref, declareOp.getShape(),
                           declareOp.getTypeparams(), declareOp.getUniqName(),
                           fortranAttrs)
                       .getResult();
    mlir::Value hlfirBase;
    mlir::Type hlfirBaseType = declareOp.getBase().getType();
    if (hlfirBaseType.isa<fir::BaseBoxType>()) {
      // Need to conditionally rebox/embox for optional.
      if (mlir::cast<fir::FortranVariableOpInterface>(declareOp.getOperation())
              .isOptional())
        TODO(loc, "converting hlfir declare of optional box to fir");
      if (!firBase.getType().isa<fir::BaseBoxType>()) {
        llvm::SmallVector<mlir::Value> typeParams;
        auto maybeCharType =
            fir::unwrapSequenceType(fir::unwrapPassByRefType(hlfirBaseType))
                .dyn_cast<fir::CharacterType>();
        if (!maybeCharType || maybeCharType.hasDynamicLen())
          typeParams.append(declareOp.getTypeparams().begin(),
                            declareOp.getTypeparams().end());
        hlfirBase = rewriter.create<fir::EmboxOp>(
            loc, hlfirBaseType, firBase, declareOp.getShape(),
            /*slice=*/mlir::Value{}, typeParams);
      } else {
        // Rebox so that lower bounds are correct.
        hlfirBase = rewriter.create<fir::ReboxOp>(loc, hlfirBaseType, firBase,
                                                  declareOp.getShape(),
                                                  /*slice=*/mlir::Value{});
      }
    } else if (hlfirBaseType.isa<fir::BoxCharType>()) {
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
public:
  explicit DesignateOpConversion(mlir::MLIRContext *ctx)
      : OpRewritePattern{ctx} {}

  mlir::LogicalResult
  matchAndRewrite(hlfir::DesignateOp designate,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = designate.getLoc();
    auto module = designate->getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, fir::getKindMapping(module));

    if (designate.getComplexPart())
      TODO(loc, "hlfir::designate with complex part");

    hlfir::Entity baseEntity(designate.getMemref());

    if (baseEntity.isMutableBox())
      TODO(loc, "hlfir::designate load of pointer or allocatable");

    mlir::Type designateResultType = designate.getResult().getType();
    llvm::SmallVector<mlir::Value> firBaseTypeParameters;
    auto [base, shape] = hlfir::genVariableFirBaseShapeAndParams(
        loc, builder, baseEntity, firBaseTypeParameters);
    mlir::Type baseEleTy = hlfir::getFortranElementType(base.getType());

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
        mlir::Type componentType = baseEleTy.cast<fir::RecordType>().getType(
            designate.getComponent().value());
        if (componentType.isa<fir::BaseBoxType>())
          TODO(loc,
               "addressing parametrized derived type automatic components");
        mlir::Type coorTy = fir::ReferenceType::get(componentType);
        base = builder.create<fir::CoordinateOp>(loc, coorTy, base, fieldIndex);
        baseEleTy = hlfir::getFortranElementType(componentType);
        shape = designate.getComponentShape();
      } else {
        // array%component[(indices) substring|complex part] cases.
        // Component ref of array bases are dealt with below in embox/rebox.
        assert(designateResultType.isa<fir::BaseBoxType>());
      }
    }

    if (designateResultType.isa<fir::BaseBoxType>()) {
      // Generate embox or rebox.
      if (!fir::unwrapPassByRefType(designateResultType)
               .isa<fir::SequenceType>())
        TODO(loc, "addressing polymorphic arrays");
      llvm::SmallVector<mlir::Value> triples;
      llvm::SmallVector<mlir::Value> sliceFields;
      mlir::Type idxTy = builder.getIndexType();
      auto subscripts = designate.getIndices();
      if (fieldIndex && baseEntity.isArray()) {
        // array%scalar_comp or array%array_comp(indices)
        // Generate triples for array(:, :, ...).
        auto one = builder.createIntegerConstant(loc, idxTy, 1);
        for (auto [lb, ub] : hlfir::genBounds(loc, builder, baseEntity)) {
          triples.push_back(builder.createConvert(loc, idxTy, lb));
          triples.push_back(builder.createConvert(loc, idxTy, ub));
          triples.push_back(one);
        }
        sliceFields.push_back(fieldIndex);
        // Add indices in the field path for "array%array_comp(indices)"
        // case.
        sliceFields.append(subscripts.begin(), subscripts.end());
      } else {
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

      mlir::Value slice;
      if (!triples.empty())
        slice =
            builder.create<fir::SliceOp>(loc, triples, sliceFields, substring);
      else
        assert(sliceFields.empty() && substring.empty());
      llvm::SmallVector<mlir::Type> resultType{designateResultType};
      mlir::Value resultBox;
      if (base.getType().isa<fir::BaseBoxType>())
        resultBox =
            builder.create<fir::ReboxOp>(loc, resultType, base, shape, slice);
      else
        resultBox = builder.create<fir::EmboxOp>(loc, resultType, base, shape,
                                                 slice, firBaseTypeParameters);
      rewriter.replaceOp(designate, resultBox);
      return mlir::success();
    }

    // Otherwise, the result is the address of a scalar, or the address of the
    // first element of a contiguous array section with compile time constant
    // shape. The base may be an array, or a scalar.
    mlir::Type resultAddressType = designateResultType;
    if (auto boxCharType = designateResultType.dyn_cast<fir::BoxCharType>())
      resultAddressType = fir::ReferenceType::get(boxCharType.getEleTy());

    // Array element indexing.
    if (!designate.getIndices().empty()) {
      // - array(indices) [substring|complex_part] or
      // - scalar%array_comp(indices) [substring|complex_part]
      // This may be a ranked contiguous array section in which case
      // The first element address is being computed.
      llvm::SmallVector<mlir::Value> firstElementIndices;
      auto indices = designate.getIndices();
      int i = 0;
      for (auto isTriplet : designate.getIsTripletAttr().asArrayRef()) {
        // Coordinate of the first element are the index and triplets lower
        // bounds
        firstElementIndices.push_back(indices[i]);
        i = i + (isTriplet ? 3 : 1);
      }
      mlir::Type arrayCoorType = fir::ReferenceType::get(baseEleTy);
      base = builder.create<fir::ArrayCoorOp>(
          loc, arrayCoorType, base, shape,
          /*slice=*/mlir::Value{}, firstElementIndices, firBaseTypeParameters);
    }

    // Scalar substring (potentially on the previously built array element or
    // component reference).
    if (!designate.getSubstring().empty())
      base = fir::factory::CharacterExprHelper{builder, loc}.genSubstringBase(
          base, designate.getSubstring()[0], resultAddressType);

    // Cast/embox the computed scalar address if needed.
    if (designateResultType.isa<fir::BoxCharType>()) {
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
};

class NoReassocOpConversion
    : public mlir::OpRewritePattern<hlfir::NoReassocOp> {
public:
  explicit NoReassocOpConversion(mlir::MLIRContext *ctx)
      : OpRewritePattern{ctx} {}

  mlir::LogicalResult
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

  mlir::LogicalResult
  matchAndRewrite(hlfir::NullOp nullop,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<fir::ZeroOp>(nullop, nullop.getType());
    return mlir::success();
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
    patterns
        .insert<AssignOpConversion, DeclareOpConversion, DesignateOpConversion,
                NoReassocOpConversion, NullOpConversion>(context);
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

std::unique_ptr<mlir::Pass> hlfir::createConvertHLFIRtoFIRPass() {
  return std::make_unique<ConvertHLFIRtoFIR>();
}
