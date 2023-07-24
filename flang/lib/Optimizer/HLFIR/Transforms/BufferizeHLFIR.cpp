//===- BufferizeHLFIR.cpp - Bufferize HLFIR  ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file defines a pass that bufferize hlfir.expr. It translates operations
// producing or consuming hlfir.expr into operations operating on memory.
// An hlfir.expr is translated to a tuple<variable address, cleanupflag>
// where cleanupflag is set to true if storage for the expression was allocated
// on the heap.
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "flang/Optimizer/Builder/Runtime/Allocatable.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

namespace hlfir {
#define GEN_PASS_DEF_BUFFERIZEHLFIR
#include "flang/Optimizer/HLFIR/Passes.h.inc"
} // namespace hlfir

namespace {

/// Helper to create tuple from a bufferized expr storage and clean up
/// instruction flag. The storage is an HLFIR variable so that it can
/// be manipulated as a variable later (all shape and length information
/// cam be retrieved from it).
static mlir::Value packageBufferizedExpr(mlir::Location loc,
                                         fir::FirOpBuilder &builder,
                                         hlfir::Entity storage,
                                         mlir::Value mustFree) {
  auto tupleType = mlir::TupleType::get(
      builder.getContext(),
      mlir::TypeRange{storage.getType(), mustFree.getType()});
  auto undef = builder.create<fir::UndefOp>(loc, tupleType);
  auto insert = builder.create<fir::InsertValueOp>(
      loc, tupleType, undef, mustFree,
      builder.getArrayAttr(
          {builder.getIntegerAttr(builder.getIndexType(), 1)}));
  return builder.create<fir::InsertValueOp>(
      loc, tupleType, insert, storage,
      builder.getArrayAttr(
          {builder.getIntegerAttr(builder.getIndexType(), 0)}));
}

/// Helper to create tuple from a bufferized expr storage and constant
/// boolean clean-up flag.
static mlir::Value packageBufferizedExpr(mlir::Location loc,
                                         fir::FirOpBuilder &builder,
                                         hlfir::Entity storage, bool mustFree) {
  mlir::Value mustFreeValue = builder.createBool(loc, mustFree);
  return packageBufferizedExpr(loc, builder, storage, mustFreeValue);
}

/// Helper to extract the storage from a tuple created by packageBufferizedExpr.
/// It assumes no tuples are used as HLFIR operation operands, which is
/// currently enforced by the verifiers that only accept HLFIR value or
/// variable types which do not include tuples.
static hlfir::Entity getBufferizedExprStorage(mlir::Value bufferizedExpr) {
  auto tupleType = bufferizedExpr.getType().dyn_cast<mlir::TupleType>();
  if (!tupleType)
    return hlfir::Entity{bufferizedExpr};
  assert(tupleType.size() == 2 && "unexpected tuple type");
  if (auto insert = bufferizedExpr.getDefiningOp<fir::InsertValueOp>())
    if (insert.getVal().getType() == tupleType.getType(0))
      return hlfir::Entity{insert.getVal()};
  TODO(bufferizedExpr.getLoc(), "general extract storage case");
}

/// Helper to extract the clean-up flag from a tuple created by
/// packageBufferizedExpr.
static mlir::Value getBufferizedExprMustFreeFlag(mlir::Value bufferizedExpr) {
  auto tupleType = bufferizedExpr.getType().dyn_cast<mlir::TupleType>();
  if (!tupleType)
    return bufferizedExpr;
  assert(tupleType.size() == 2 && "unexpected tuple type");
  if (auto insert = bufferizedExpr.getDefiningOp<fir::InsertValueOp>())
    if (auto insert0 = insert.getAdt().getDefiningOp<fir::InsertValueOp>())
      if (insert0.getVal().getType() == tupleType.getType(1))
        return insert0.getVal();
  TODO(bufferizedExpr.getLoc(), "general extract storage case");
}

static std::pair<hlfir::Entity, mlir::Value>
createTempFromMold(mlir::Location loc, fir::FirOpBuilder &builder,
                   hlfir::Entity mold) {
  llvm::SmallVector<mlir::Value> lenParams;
  hlfir::genLengthParameters(loc, builder, mold, lenParams);
  llvm::StringRef tmpName{".tmp"};
  mlir::Value alloc;
  mlir::Value isHeapAlloc;
  mlir::Value shape{};
  fir::FortranVariableFlagsAttr declAttrs;

  if (mold.isPolymorphic()) {
    // Create unallocated polymorphic temporary using the dynamic type
    // of the mold. The static type of the temporary matches
    // the static type of the mold, but then the dynamic type
    // of the mold is applied to the temporary's descriptor.

    if (mold.isArray())
      hlfir::genShape(loc, builder, mold);

    // Create polymorphic allocatable box on the stack.
    mlir::Type boxHeapType = fir::HeapType::get(fir::unwrapRefType(
        mlir::cast<fir::BaseBoxType>(mold.getType()).getEleTy()));
    // The box must be initialized, because AllocatableApplyMold
    // may read its contents (e.g. for checking whether it is allocated).
    alloc = fir::factory::genNullBoxStorage(builder, loc,
                                            fir::ClassType::get(boxHeapType));
    // The temporary is unallocated even after AllocatableApplyMold below.
    // If the temporary is used as assignment LHS it will be automatically
    // allocated on the heap, as long as we use Assign family
    // runtime functions. So set MustFree to true.
    isHeapAlloc = builder.createBool(loc, true);
    declAttrs = fir::FortranVariableFlagsAttr::get(
        builder.getContext(), fir::FortranVariableFlagsEnum::allocatable);
  } else if (mold.isArray()) {
    mlir::Type sequenceType =
        hlfir::getFortranElementOrSequenceType(mold.getType());
    shape = hlfir::genShape(loc, builder, mold);
    auto extents = hlfir::getIndexExtents(loc, builder, shape);
    alloc = builder.createHeapTemporary(loc, sequenceType, tmpName, extents,
                                        lenParams);
    isHeapAlloc = builder.createBool(loc, true);
  } else {
    alloc = builder.createTemporary(loc, mold.getFortranElementType(), tmpName,
                                    /*shape=*/std::nullopt, lenParams);
    isHeapAlloc = builder.createBool(loc, false);
  }
  auto declareOp = builder.create<hlfir::DeclareOp>(loc, alloc, tmpName, shape,
                                                    lenParams, declAttrs);
  if (mold.isPolymorphic()) {
    int rank = mold.getRank();
    // TODO: should probably read rank from the mold.
    if (rank < 0)
      TODO(loc, "create temporary for assumed rank polymorphic");
    fir::runtime::genAllocatableApplyMold(builder, loc, alloc,
                                          mold.getFirBase(), rank);
  }

  return {hlfir::Entity{declareOp.getBase()}, isHeapAlloc};
}

static std::pair<hlfir::Entity, mlir::Value>
createArrayTemp(mlir::Location loc, fir::FirOpBuilder &builder,
                mlir::Type exprType, mlir::Value shape,
                mlir::ValueRange extents, mlir::ValueRange lenParams) {
  mlir::Type sequenceType = hlfir::getFortranElementOrSequenceType(exprType);
  llvm::StringRef tmpName{".tmp.array"};
  mlir::Value allocmem = builder.createHeapTemporary(loc, sequenceType, tmpName,
                                                     extents, lenParams);
  auto declareOp =
      builder.create<hlfir::DeclareOp>(loc, allocmem, tmpName, shape, lenParams,
                                       fir::FortranVariableFlagsAttr{});
  mlir::Value trueVal = builder.createBool(loc, true);
  return {hlfir::Entity{declareOp.getBase()}, trueVal};
}

struct AsExprOpConversion : public mlir::OpConversionPattern<hlfir::AsExprOp> {
  using mlir::OpConversionPattern<hlfir::AsExprOp>::OpConversionPattern;
  explicit AsExprOpConversion(mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<hlfir::AsExprOp>{ctx} {}
  mlir::LogicalResult
  matchAndRewrite(hlfir::AsExprOp asExpr, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = asExpr->getLoc();
    auto module = asExpr->getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, module);
    if (asExpr.isMove()) {
      // Move variable storage for the hlfir.expr buffer.
      mlir::Value bufferizedExpr = packageBufferizedExpr(
          loc, builder, hlfir::Entity{adaptor.getVar()}, adaptor.getMustFree());
      rewriter.replaceOp(asExpr, bufferizedExpr);
      return mlir::success();
    }
    // Otherwise, create a copy in a new buffer.
    hlfir::Entity source = hlfir::Entity{adaptor.getVar()};
    auto [temp, cleanup] = createTempFromMold(loc, builder, source);
    builder.create<hlfir::AssignOp>(loc, source, temp, temp.isAllocatable(),
                                    /*keep_lhs_length_if_realloc=*/false,
                                    /*temporary_lhs=*/true);
    mlir::Value bufferizedExpr =
        packageBufferizedExpr(loc, builder, temp, cleanup);
    rewriter.replaceOp(asExpr, bufferizedExpr);
    return mlir::success();
  }
};

struct ShapeOfOpConversion
    : public mlir::OpConversionPattern<hlfir::ShapeOfOp> {
  using mlir::OpConversionPattern<hlfir::ShapeOfOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(hlfir::ShapeOfOp shapeOf, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = shapeOf.getLoc();
    mlir::ModuleOp mod = shapeOf->getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);

    mlir::Value shape;
    hlfir::Entity bufferizedExpr{getBufferizedExprStorage(adaptor.getExpr())};
    if (bufferizedExpr.isVariable()) {
      shape = hlfir::genShape(loc, builder, bufferizedExpr);
    } else {
      // everything else failed so try to create a shape from static type info
      hlfir::ExprType exprTy =
          adaptor.getExpr().getType().dyn_cast_or_null<hlfir::ExprType>();
      if (exprTy)
        shape = hlfir::genExprShape(builder, loc, exprTy);
    }
    // expected to never happen
    if (!shape)
      return emitError(loc,
                       "Unresolvable hlfir.shape_of where extents are unknown");

    rewriter.replaceOp(shapeOf, shape);
    return mlir::success();
  }
};

struct ApplyOpConversion : public mlir::OpConversionPattern<hlfir::ApplyOp> {
  using mlir::OpConversionPattern<hlfir::ApplyOp>::OpConversionPattern;
  explicit ApplyOpConversion(mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<hlfir::ApplyOp>{ctx} {}
  mlir::LogicalResult
  matchAndRewrite(hlfir::ApplyOp apply, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = apply->getLoc();
    hlfir::Entity bufferizedExpr = getBufferizedExprStorage(adaptor.getExpr());
    mlir::Type resultType = hlfir::getVariableElementType(bufferizedExpr);
    mlir::Value result = rewriter.create<hlfir::DesignateOp>(
        loc, resultType, bufferizedExpr, adaptor.getIndices(),
        adaptor.getTypeparams());
    if (fir::isa_trivial(apply.getType())) {
      result = rewriter.create<fir::LoadOp>(loc, result);
    } else {
      fir::FirOpBuilder builder(rewriter, apply.getOperation());
      result =
          packageBufferizedExpr(loc, builder, hlfir::Entity{result}, false);
    }
    rewriter.replaceOp(apply, result);
    return mlir::success();
  }
};

struct AssignOpConversion : public mlir::OpConversionPattern<hlfir::AssignOp> {
  using mlir::OpConversionPattern<hlfir::AssignOp>::OpConversionPattern;
  explicit AssignOpConversion(mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<hlfir::AssignOp>{ctx} {}
  mlir::LogicalResult
  matchAndRewrite(hlfir::AssignOp assign, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Value> newOperands;
    for (mlir::Value operand : adaptor.getOperands())
      newOperands.push_back(getBufferizedExprStorage(operand));
    rewriter.startRootUpdate(assign);
    assign->setOperands(newOperands);
    rewriter.finalizeRootUpdate(assign);
    return mlir::success();
  }
};

struct ConcatOpConversion : public mlir::OpConversionPattern<hlfir::ConcatOp> {
  using mlir::OpConversionPattern<hlfir::ConcatOp>::OpConversionPattern;
  explicit ConcatOpConversion(mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<hlfir::ConcatOp>{ctx} {}
  mlir::LogicalResult
  matchAndRewrite(hlfir::ConcatOp concat, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = concat->getLoc();
    fir::FirOpBuilder builder(rewriter, concat.getOperation());
    assert(adaptor.getStrings().size() >= 2 &&
           "must have at least two strings operands");
    if (adaptor.getStrings().size() > 2)
      TODO(loc, "codegen of optimized chained concatenation of more than two "
                "strings");
    hlfir::Entity lhs = getBufferizedExprStorage(adaptor.getStrings()[0]);
    hlfir::Entity rhs = getBufferizedExprStorage(adaptor.getStrings()[1]);
    auto [lhsExv, c1] = hlfir::translateToExtendedValue(loc, builder, lhs);
    auto [rhsExv, c2] = hlfir::translateToExtendedValue(loc, builder, rhs);
    assert(!c1 && !c2 && "expected variables");
    fir::ExtendedValue res =
        fir::factory::CharacterExprHelper{builder, loc}.createConcatenate(
            *lhsExv.getCharBox(), *rhsExv.getCharBox());
    // Ensure the memory type is the same as the result type.
    mlir::Type addrType = fir::ReferenceType::get(
        hlfir::getFortranElementType(concat.getResult().getType()));
    mlir::Value cast = builder.createConvert(loc, addrType, fir::getBase(res));
    res = fir::substBase(res, cast);
    hlfir::Entity hlfirTempRes =
        hlfir::Entity{hlfir::genDeclare(loc, builder, res, "tmp",
                                        fir::FortranVariableFlagsAttr{})
                          .getBase()};
    mlir::Value bufferizedExpr =
        packageBufferizedExpr(loc, builder, hlfirTempRes, false);
    rewriter.replaceOp(concat, bufferizedExpr);
    return mlir::success();
  }
};

struct SetLengthOpConversion
    : public mlir::OpConversionPattern<hlfir::SetLengthOp> {
  using mlir::OpConversionPattern<hlfir::SetLengthOp>::OpConversionPattern;
  explicit SetLengthOpConversion(mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<hlfir::SetLengthOp>{ctx} {}
  mlir::LogicalResult
  matchAndRewrite(hlfir::SetLengthOp setLength, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = setLength->getLoc();
    fir::FirOpBuilder builder(rewriter, setLength.getOperation());
    // Create a temp with the new length.
    hlfir::Entity string = getBufferizedExprStorage(adaptor.getString());
    auto charType = hlfir::getFortranElementType(setLength.getType());
    llvm::StringRef tmpName{".tmp"};
    llvm::SmallVector<mlir::Value, 1> lenParams{adaptor.getLength()};
    auto alloca = builder.createTemporary(loc, charType, tmpName,
                                          /*shape=*/std::nullopt, lenParams);
    auto declareOp = builder.create<hlfir::DeclareOp>(
        loc, alloca, tmpName, /*shape=*/mlir::Value{}, lenParams,
        fir::FortranVariableFlagsAttr{});
    hlfir::Entity temp{declareOp.getBase()};
    // Assign string value to the created temp.
    builder.create<hlfir::AssignOp>(loc, string, temp,
                                    /*realloc=*/false,
                                    /*keep_lhs_length_if_realloc=*/false,
                                    /*temporary_lhs=*/true);
    mlir::Value bufferizedExpr =
        packageBufferizedExpr(loc, builder, temp, false);
    rewriter.replaceOp(setLength, bufferizedExpr);
    return mlir::success();
  }
};

struct GetLengthOpConversion
    : public mlir::OpConversionPattern<hlfir::GetLengthOp> {
  using mlir::OpConversionPattern<hlfir::GetLengthOp>::OpConversionPattern;
  explicit GetLengthOpConversion(mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<hlfir::GetLengthOp>{ctx} {}
  mlir::LogicalResult
  matchAndRewrite(hlfir::GetLengthOp getLength, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = getLength->getLoc();
    fir::FirOpBuilder builder(rewriter, getLength.getOperation());
    hlfir::Entity bufferizedExpr = getBufferizedExprStorage(adaptor.getExpr());
    mlir::Value length = hlfir::genCharLength(loc, builder, bufferizedExpr);
    if (!length)
      return rewriter.notifyMatchFailure(
          getLength, "could not deduce length from GetLengthOp operand");
    rewriter.replaceOp(getLength, length);
    return mlir::success();
  }
};

/// The current hlfir.associate lowering does not handle multiple uses of a
/// non-trivial expression value because it generates the cleanup for the
/// expression bufferization at hlfir.end_associate. If there was more than one
/// hlfir.end_associate, it would be cleaned up multiple times, perhaps before
/// one of the other uses.
/// Note that we have to be careful about expressions used by a single
/// hlfir.end_associate that may be executed more times than the producer
/// of the expression value. This may also cause multiple clean-ups
/// for the same memory (e.g. cause double-free errors). For example,
/// hlfir.end_associate inside hlfir.elemental may cause such issues
/// for expressions produced outside of hlfir.elemental.
static bool allOtherUsesAreSafeForAssociate(mlir::Value value,
                                            mlir::Operation *currentUse,
                                            mlir::Operation *endAssociate) {
  // If value producer is from a different region than
  // hlfir.associate/end_associate, then conservatively assume
  // that the hlfir.end_associate may execute more times than
  // the value producer.
  // TODO: this may be improved for operations that cannot
  // result in multiple executions (e.g. ifOp).
  if (value.getParentRegion() != currentUse->getParentRegion() ||
      (endAssociate &&
       value.getParentRegion() != endAssociate->getParentRegion()))
    return false;

  for (mlir::Operation *useOp : value.getUsers())
    if (!mlir::isa<hlfir::DestroyOp>(useOp) && useOp != currentUse) {
      // hlfir.shape_of and hlfir.get_length will not disrupt cleanup so it is
      // safe for hlfir.associate. These operations might read from the box and
      // so they need to come before the hflir.end_associate (which may
      // deallocate).
      if (mlir::isa<hlfir::ShapeOfOp>(useOp) ||
          mlir::isa<hlfir::GetLengthOp>(useOp)) {
        if (!endAssociate)
          continue;
        // not known to occur in practice:
        if (useOp->getBlock() != endAssociate->getBlock())
          TODO(endAssociate->getLoc(), "Associate split over multiple blocks");
        if (useOp->isBeforeInBlock(endAssociate))
          continue;
      }
      return false;
    }
  return true;
}

static void eraseAllUsesInDestroys(mlir::Value value,
                                   mlir::ConversionPatternRewriter &rewriter) {
  for (mlir::Operation *useOp : value.getUsers())
    if (mlir::isa<hlfir::DestroyOp>(useOp))
      rewriter.eraseOp(useOp);
}

struct AssociateOpConversion
    : public mlir::OpConversionPattern<hlfir::AssociateOp> {
  using mlir::OpConversionPattern<hlfir::AssociateOp>::OpConversionPattern;
  explicit AssociateOpConversion(mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<hlfir::AssociateOp>{ctx} {}
  mlir::LogicalResult
  matchAndRewrite(hlfir::AssociateOp associate, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = associate->getLoc();
    fir::FirOpBuilder builder(rewriter, associate.getOperation());
    mlir::Value bufferizedExpr = getBufferizedExprStorage(adaptor.getSource());
    const bool isTrivialValue = fir::isa_trivial(bufferizedExpr.getType());

    auto getEndAssociate =
        [](hlfir::AssociateOp associate) -> mlir::Operation * {
      for (mlir::Operation *useOp : associate->getUsers())
        if (mlir::isa<hlfir::EndAssociateOp>(useOp))
          return useOp;
      // happens in some hand coded mlir in tests
      return nullptr;
    };

    auto replaceWith = [&](mlir::Value hlfirVar, mlir::Value firVar,
                           mlir::Value flag) {
      // 0-dim variables may need special handling:
      //   %0 = hlfir.as_expr %x move %true :
      //       (!fir.box<!fir.heap<!fir.type<_T{y:i32}>>>, i1) ->
      //       !hlfir.expr<!fir.type<_T{y:i32}>>
      //   %1:3 = hlfir.associate %0 {uniq_name = "adapt.valuebyref"} :
      //       (!hlfir.expr<!fir.type<_T{y:i32}>>) ->
      //       (!fir.ref<!fir.type<_T{y:i32}>>,
      //        !fir.ref<!fir.type<_T{y:i32}>>,
      //        i1)
      //
      // !fir.box<!fir.heap<!fir.type<_T{y:i32}>>> value must be
      // propagated as the box address !fir.ref<!fir.type<_T{y:i32}>>.
      auto adjustVar = [&](mlir::Value sourceVar, mlir::Type assocType) {
        if (mlir::isa<fir::ReferenceType>(sourceVar.getType()) &&
            mlir::isa<fir::ClassType>(
                fir::unwrapRefType(sourceVar.getType()))) {
          // Association of a polymorphic value.
          sourceVar = builder.create<fir::LoadOp>(loc, sourceVar);
          assert(mlir::isa<fir::ClassType>(sourceVar.getType()) &&
                 fir::isAllocatableType(sourceVar.getType()));
          assert(sourceVar.getType() == assocType);
        } else if ((sourceVar.getType().isa<fir::BaseBoxType>() &&
                    !assocType.isa<fir::BaseBoxType>()) ||
                   ((sourceVar.getType().isa<fir::BoxCharType>() &&
                     !assocType.isa<fir::BoxCharType>()))) {
          sourceVar = builder.create<fir::BoxAddrOp>(loc, assocType, sourceVar);
        } else {
          sourceVar = builder.createConvert(loc, assocType, sourceVar);
        }
        return sourceVar;
      };

      mlir::Type associateHlfirVarType = associate.getResultTypes()[0];
      hlfirVar = adjustVar(hlfirVar, associateHlfirVarType);
      associate.getResult(0).replaceAllUsesWith(hlfirVar);

      mlir::Type associateFirVarType = associate.getResultTypes()[1];
      firVar = adjustVar(firVar, associateFirVarType);
      associate.getResult(1).replaceAllUsesWith(firVar);
      associate.getResult(2).replaceAllUsesWith(flag);
      rewriter.replaceOp(associate, {hlfirVar, firVar, flag});
    };

    // If this is the last use of the expression value and this is an hlfir.expr
    // that was bufferized, re-use the storage.
    // Otherwise, create a temp and assign the storage to it.
    if (!isTrivialValue && allOtherUsesAreSafeForAssociate(
                               adaptor.getSource(), associate.getOperation(),
                               getEndAssociate(associate))) {
      // Re-use hlfir.expr buffer if this is the only use of the hlfir.expr
      // outside of the hlfir.destroy. Take on the cleaning-up responsibility
      // for the related hlfir.end_associate, and erase the hlfir.destroy (if
      // any).
      mlir::Value mustFree = getBufferizedExprMustFreeFlag(adaptor.getSource());
      mlir::Value firBase = hlfir::Entity{bufferizedExpr}.getFirBase();
      replaceWith(bufferizedExpr, firBase, mustFree);
      eraseAllUsesInDestroys(associate.getSource(), rewriter);
      return mlir::success();
    }
    if (isTrivialValue) {
      auto temp = builder.createTemporary(loc, bufferizedExpr.getType(),
                                          associate.getUniqName());
      builder.create<fir::StoreOp>(loc, bufferizedExpr, temp);
      mlir::Value mustFree = builder.createBool(loc, false);
      replaceWith(temp, temp, mustFree);
      return mlir::success();
    }
    // non-trivial value with more than one use. We will have to make a copy and
    // use that
    hlfir::Entity source = hlfir::Entity{bufferizedExpr};
    auto [temp, cleanup] = createTempFromMold(loc, builder, source);
    builder.create<hlfir::AssignOp>(loc, source, temp, temp.isAllocatable(),
                                    /*keep_lhs_length_if_realloc=*/false,
                                    /*temporary_lhs=*/true);
    mlir::Value bufferTuple =
        packageBufferizedExpr(loc, builder, temp, cleanup);
    bufferizedExpr = getBufferizedExprStorage(bufferTuple);
    replaceWith(bufferizedExpr, hlfir::Entity{bufferizedExpr}.getFirBase(),
                getBufferizedExprMustFreeFlag(bufferTuple));
    return mlir::success();
  }
};

static void genFreeIfMustFree(mlir::Location loc, fir::FirOpBuilder &builder,
                              mlir::Value var, mlir::Value mustFree) {
  auto genFree = [&]() {
    // fir::FreeMemOp operand type must be a fir::HeapType.
    mlir::Type heapType = fir::HeapType::get(
        hlfir::getFortranElementOrSequenceType(var.getType()));
    if (mlir::isa<fir::ReferenceType>(var.getType()) &&
        mlir::isa<fir::ClassType>(fir::unwrapRefType(var.getType()))) {
      // A temporary for a polymorphic expression is represented
      // via an allocatable. Variable type in this case
      // is !fir.ref<!fir.class<!fir.heap<!fir.type<>>>>.
      // We need to free the allocatable data, not the box
      // that is allocated on the stack.
      var = builder.create<fir::LoadOp>(loc, var);
      assert(mlir::isa<fir::ClassType>(var.getType()) &&
             fir::isAllocatableType(var.getType()));
      var = builder.create<fir::BoxAddrOp>(loc, heapType, var);
    } else if (var.getType().isa<fir::BaseBoxType, fir::BoxCharType>()) {
      var = builder.create<fir::BoxAddrOp>(loc, heapType, var);
    } else if (!var.getType().isa<fir::HeapType>()) {
      var = builder.create<fir::ConvertOp>(loc, heapType, var);
    }
    builder.create<fir::FreeMemOp>(loc, var);
  };
  if (auto cstMustFree = fir::getIntIfConstant(mustFree)) {
    if (*cstMustFree != 0)
      genFree();
    // else, mustFree is false, nothing to do.
    return;
  }
  builder.genIfThen(loc, mustFree).genThen(genFree).end();
}

struct EndAssociateOpConversion
    : public mlir::OpConversionPattern<hlfir::EndAssociateOp> {
  using mlir::OpConversionPattern<hlfir::EndAssociateOp>::OpConversionPattern;
  explicit EndAssociateOpConversion(mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<hlfir::EndAssociateOp>{ctx} {}
  mlir::LogicalResult
  matchAndRewrite(hlfir::EndAssociateOp endAssociate, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = endAssociate->getLoc();
    fir::FirOpBuilder builder(rewriter, endAssociate.getOperation());
    genFreeIfMustFree(loc, builder, adaptor.getVar(), adaptor.getMustFree());
    rewriter.eraseOp(endAssociate);
    return mlir::success();
  }
};

struct DestroyOpConversion
    : public mlir::OpConversionPattern<hlfir::DestroyOp> {
  using mlir::OpConversionPattern<hlfir::DestroyOp>::OpConversionPattern;
  explicit DestroyOpConversion(mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<hlfir::DestroyOp>{ctx} {}
  mlir::LogicalResult
  matchAndRewrite(hlfir::DestroyOp destroy, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // If expr was bufferized on the heap, now is time to deallocate the buffer.
    mlir::Location loc = destroy->getLoc();
    hlfir::Entity bufferizedExpr = getBufferizedExprStorage(adaptor.getExpr());
    if (!fir::isa_trivial(bufferizedExpr.getType())) {
      fir::FirOpBuilder builder(rewriter, destroy.getOperation());
      mlir::Value mustFree = getBufferizedExprMustFreeFlag(adaptor.getExpr());
      mlir::Value firBase = bufferizedExpr.getFirBase();
      genFreeIfMustFree(loc, builder, firBase, mustFree);
    }
    rewriter.eraseOp(destroy);
    return mlir::success();
  }
};

struct NoReassocOpConversion
    : public mlir::OpConversionPattern<hlfir::NoReassocOp> {
  using mlir::OpConversionPattern<hlfir::NoReassocOp>::OpConversionPattern;
  explicit NoReassocOpConversion(mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<hlfir::NoReassocOp>{ctx} {}
  mlir::LogicalResult
  matchAndRewrite(hlfir::NoReassocOp noreassoc, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = noreassoc->getLoc();
    fir::FirOpBuilder builder(rewriter, noreassoc.getOperation());
    mlir::Value bufferizedExpr = getBufferizedExprStorage(adaptor.getVal());
    mlir::Value result =
        builder.create<hlfir::NoReassocOp>(loc, bufferizedExpr);

    if (!fir::isa_trivial(bufferizedExpr.getType())) {
      // NoReassocOp should not be needed on the mustFree path.
      mlir::Value mustFree = getBufferizedExprMustFreeFlag(adaptor.getVal());
      result =
          packageBufferizedExpr(loc, builder, hlfir::Entity{result}, mustFree);
    }
    rewriter.replaceOp(noreassoc, result);
    return mlir::success();
  }
};

/// Was \p value created in the mlir block where \p builder is currently set ?
static bool wasCreatedInCurrentBlock(mlir::Value value,
                                     fir::FirOpBuilder &builder) {
  if (mlir::Operation *op = value.getDefiningOp())
    return op->getBlock() == builder.getBlock();
  return false;
}

/// This Listener allows setting both the builder and the rewriter as
/// listeners. This is required when a pattern uses a firBuilder helper that
/// may create illegal operations that will need to be translated and requires
/// notifying the rewriter.
struct HLFIRListener : public mlir::OpBuilder::Listener {
  HLFIRListener(fir::FirOpBuilder &builder,
                mlir::ConversionPatternRewriter &rewriter)
      : builder{builder}, rewriter{rewriter} {}
  void notifyOperationInserted(mlir::Operation *op) override {
    builder.notifyOperationInserted(op);
    rewriter.notifyOperationInserted(op);
  }
  virtual void notifyBlockCreated(mlir::Block *block) override {
    builder.notifyBlockCreated(block);
    rewriter.notifyBlockCreated(block);
  }
  fir::FirOpBuilder &builder;
  mlir::ConversionPatternRewriter &rewriter;
};

struct ElementalOpConversion
    : public mlir::OpConversionPattern<hlfir::ElementalOp> {
  using mlir::OpConversionPattern<hlfir::ElementalOp>::OpConversionPattern;
  explicit ElementalOpConversion(mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<hlfir::ElementalOp>{ctx} {
    // This pattern recursively converts nested ElementalOp's
    // by cloning and then converting them, so we have to allow
    // for recursive pattern application. The recursion is bounded
    // by the nesting level of ElementalOp's.
    setHasBoundedRewriteRecursion();
  }
  mlir::LogicalResult
  matchAndRewrite(hlfir::ElementalOp elemental, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = elemental->getLoc();
    fir::FirOpBuilder builder(rewriter, elemental.getOperation());
    // The body of the elemental op may contain operation that will require
    // to be translated. Notify the rewriter about the cloned operations.
    HLFIRListener listener{builder, rewriter};
    builder.setListener(&listener);

    mlir::Value shape = adaptor.getShape();
    auto extents = hlfir::getIndexExtents(loc, builder, shape);
    auto [temp, cleanup] =
        createArrayTemp(loc, builder, elemental.getType(), shape, extents,
                        adaptor.getTypeparams());
    // Generate a loop nest looping around the fir.elemental shape and clone
    // fir.elemental region inside the inner loop.
    hlfir::LoopNest loopNest =
        hlfir::genLoopNest(loc, builder, extents, !elemental.isOrdered());
    auto insPt = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(loopNest.innerLoop.getBody());
    auto yield = hlfir::inlineElementalOp(loc, builder, elemental,
                                          loopNest.oneBasedIndices);
    hlfir::Entity elementValue(yield.getElementValue());
    // Skip final AsExpr if any. It would create an element temporary,
    // which is no needed since the element will be assigned right away in
    // the array temporary. An hlfir.as_expr may have been added if the
    // elemental is a "view" over a variable (e.g parentheses or transpose).
    if (auto asExpr = elementValue.getDefiningOp<hlfir::AsExprOp>()) {
      if (asExpr->hasOneUse() && !asExpr.isMove()) {
        elementValue = hlfir::Entity{asExpr.getVar()};
        rewriter.eraseOp(asExpr);
      }
    }
    rewriter.eraseOp(yield);
    // Assign the element value to the temp element for this iteration.
    auto tempElement =
        hlfir::getElementAt(loc, builder, temp, loopNest.oneBasedIndices);
    builder.create<hlfir::AssignOp>(loc, elementValue, tempElement,
                                    /*realloc=*/false,
                                    /*keep_lhs_length_if_realloc=*/false,
                                    /*temporary_lhs=*/true);
    // hlfir.yield_element implicitly marks the end-of-life its operand if
    // it is an expression created in the hlfir.elemental (since it is its
    // last use and an hlfir.destroy could not be created afterwards)
    // Now that this node has been removed and the expression has been used in
    // the assign, insert an hlfir.destroy to mark the expression end-of-life.
    // If the expression creation allocated a buffer on the heap inside the
    // loop, this will ensure the buffer properly deallocated.
    if (elementValue.getType().isa<hlfir::ExprType>() &&
        wasCreatedInCurrentBlock(elementValue, builder))
      builder.create<hlfir::DestroyOp>(loc, elementValue);
    builder.restoreInsertionPoint(insPt);

    mlir::Value bufferizedExpr =
        packageBufferizedExpr(loc, builder, temp, cleanup);
    rewriter.replaceOp(elemental, bufferizedExpr);
    return mlir::success();
  }
};
struct CharExtremumOpConversion
    : public mlir::OpConversionPattern<hlfir::CharExtremumOp> {
  using mlir::OpConversionPattern<hlfir::CharExtremumOp>::OpConversionPattern;
  explicit CharExtremumOpConversion(mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<hlfir::CharExtremumOp>{ctx} {}
  mlir::LogicalResult
  matchAndRewrite(hlfir::CharExtremumOp char_extremum, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = char_extremum->getLoc();
    auto predicate = char_extremum.getPredicate();
    bool predIsMin =
        predicate == hlfir::CharExtremumPredicate::min ? true : false;
    fir::FirOpBuilder builder(rewriter, char_extremum.getOperation());
    assert(adaptor.getStrings().size() >= 2 &&
           "must have at least two strings operands");
    auto numOperands = adaptor.getStrings().size();

    std::vector<hlfir::Entity> chars;
    std::vector<
        std::pair<fir::ExtendedValue, std::optional<hlfir::CleanupFunction>>>
        pairs;
    llvm::SmallVector<fir::CharBoxValue> opCBVs;
    for (size_t i = 0; i < numOperands; ++i) {
      chars.emplace_back(getBufferizedExprStorage(adaptor.getStrings()[i]));
      pairs.emplace_back(
          hlfir::translateToExtendedValue(loc, builder, chars[i]));
      assert(!pairs[i].second && "expected variables");
      opCBVs.emplace_back(*pairs[i].first.getCharBox());
    }

    fir::ExtendedValue res =
        fir::factory::CharacterExprHelper{builder, loc}.createCharExtremum(
            predIsMin, opCBVs);
    mlir::Type addrType = fir::ReferenceType::get(
        hlfir::getFortranElementType(char_extremum.getResult().getType()));
    mlir::Value cast = builder.createConvert(loc, addrType, fir::getBase(res));
    res = fir::substBase(res, cast);
    hlfir::Entity hlfirTempRes =
        hlfir::Entity{hlfir::genDeclare(loc, builder, res, ".tmp.char_extremum",
                                        fir::FortranVariableFlagsAttr{})
                          .getBase()};
    mlir::Value bufferizedExpr =
        packageBufferizedExpr(loc, builder, hlfirTempRes, false);
    rewriter.replaceOp(char_extremum, bufferizedExpr);
    return mlir::success();
  }
};

class BufferizeHLFIR : public hlfir::impl::BufferizeHLFIRBase<BufferizeHLFIR> {
public:
  void runOnOperation() override {
    // TODO: make this a pass operating on FuncOp. The issue is that
    // FirOpBuilder helpers may generate new FuncOp because of runtime/llvm
    // intrinsics calls creation. This may create race conflict if the pass is
    // scheduled on FuncOp. A solution could be to provide an optional mutex
    // when building a FirOpBuilder and locking around FuncOp and GlobalOp
    // creation, but this needs a bit more thinking, so at this point the pass
    // is scheduled on the moduleOp.
    auto module = this->getOperation();
    auto *context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.insert<ApplyOpConversion, AsExprOpConversion, AssignOpConversion,
                    AssociateOpConversion, CharExtremumOpConversion,
                    ConcatOpConversion, DestroyOpConversion,
                    ElementalOpConversion, EndAssociateOpConversion,
                    NoReassocOpConversion, SetLengthOpConversion,
                    ShapeOfOpConversion, GetLengthOpConversion>(context);
    mlir::ConversionTarget target(*context);
    // Note that YieldElementOp is not marked as an illegal operation.
    // It must be erased by its parent converter and there is no explicit
    // conversion pattern to YieldElementOp itself. If any YieldElementOp
    // survives this pass, the verifier will detect it because it has to be
    // a child of ElementalOp and ElementalOp's are explicitly illegal.
    target.addIllegalOp<hlfir::ApplyOp, hlfir::AssociateOp, hlfir::ElementalOp,
                        hlfir::EndAssociateOp, hlfir::SetLengthOp>();

    target.markUnknownOpDynamicallyLegal([](mlir::Operation *op) {
      return llvm::all_of(
                 op->getResultTypes(),
                 [](mlir::Type ty) { return !ty.isa<hlfir::ExprType>(); }) &&
             llvm::all_of(op->getOperandTypes(), [](mlir::Type ty) {
               return !ty.isa<hlfir::ExprType>();
             });
    });
    if (mlir::failed(
            mlir::applyFullConversion(module, target, std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(context),
                      "failure in HLFIR bufferization pass");
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> hlfir::createBufferizeHLFIRPass() {
  return std::make_unique<BufferizeHLFIR>();
}
