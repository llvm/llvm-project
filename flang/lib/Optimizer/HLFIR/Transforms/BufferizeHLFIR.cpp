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
#include "flang/Optimizer/Builder/Runtime/Derived.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "mlir/IR/Dominance.h"
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
createArrayTemp(mlir::Location loc, fir::FirOpBuilder &builder,
                mlir::Type exprType, mlir::Value shape,
                mlir::ValueRange extents, mlir::ValueRange lenParams,
                std::optional<hlfir::Entity> polymorphicMold) {
  mlir::Type sequenceType = hlfir::getFortranElementOrSequenceType(exprType);
  llvm::StringRef tmpName{".tmp.array"};

  if (polymorphicMold) {
    // Create *allocated* polymorphic temporary using the dynamic type
    // of the mold and the provided shape/extents. The created temporary
    // array will be written element per element, that is why it has to be
    // allocated.
    mlir::Type boxHeapType = fir::HeapType::get(sequenceType);
    mlir::Value alloc = fir::factory::genNullBoxStorage(
        builder, loc, fir::ClassType::get(boxHeapType));
    mlir::Value isHeapAlloc = builder.createBool(loc, true);
    fir::FortranVariableFlagsAttr declAttrs =
        fir::FortranVariableFlagsAttr::get(
            builder.getContext(), fir::FortranVariableFlagsEnum::allocatable);

    auto declareOp = builder.create<hlfir::DeclareOp>(loc, alloc, tmpName,
                                                      /*shape=*/nullptr,
                                                      lenParams, declAttrs);

    int rank = extents.size();
    fir::runtime::genAllocatableApplyMold(builder, loc, alloc,
                                          polymorphicMold->getFirBase(), rank);
    if (!extents.empty()) {
      mlir::Type idxTy = builder.getIndexType();
      mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
      unsigned dim = 0;
      for (mlir::Value extent : extents) {
        mlir::Value dimIndex = builder.createIntegerConstant(loc, idxTy, dim++);
        fir::runtime::genAllocatableSetBounds(builder, loc, alloc, dimIndex,
                                              one, extent);
      }
    }
    if (!lenParams.empty()) {
      // We should call AllocatableSetDerivedLength() here.
      // TODO: does the mold provide the length parameters or
      // the operation itself or should they be in sync?
      TODO(loc, "polymorphic type with length parameters in HLFIR");
    }
    fir::runtime::genAllocatableAllocate(builder, loc, alloc);

    return {hlfir::Entity{declareOp.getBase()}, isHeapAlloc};
  }

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
    auto [temp, cleanup] = hlfir::createTempFromMold(loc, builder, source);
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

  for (mlir::Operation *useOp : value.getUsers()) {
    // Ignore DestroyOp's that do not imply finalization.
    // If finalization is implied, then we must delegate
    // the finalization to the correspoding EndAssociateOp,
    // but we currently do not; so we disable the buffer
    // reuse in this case.
    if (auto destroy = mlir::dyn_cast<hlfir::DestroyOp>(useOp)) {
      if (destroy.mustFinalizeExpr())
        return false;
      else
        continue;
    }

    if (useOp != currentUse) {
      // hlfir.shape_of and hlfir.get_length will not disrupt cleanup so it is
      // safe for hlfir.associate. These operations might read from the box and
      // so they need to come before the hflir.end_associate (which may
      // deallocate).
      if (mlir::isa<hlfir::ShapeOfOp>(useOp) ||
          mlir::isa<hlfir::GetLengthOp>(useOp)) {
        if (!endAssociate)
          continue;
        // If useOp dominates the endAssociate, then it is definitely safe.
        if (useOp->getBlock() != endAssociate->getBlock())
          if (mlir::DominanceInfo{}.dominates(useOp, endAssociate))
            continue;
        if (useOp->isBeforeInBlock(endAssociate))
          continue;
      }
      return false;
    }
  }
  return true;
}

static void eraseAllUsesInDestroys(mlir::Value value,
                                   mlir::ConversionPatternRewriter &rewriter) {
  for (mlir::Operation *useOp : value.getUsers())
    if (auto destroy = mlir::dyn_cast<hlfir::DestroyOp>(useOp)) {
      assert(!destroy.mustFinalizeExpr() &&
             "deleting DestroyOp with finalize attribute");
      rewriter.eraseOp(destroy);
    }
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
      //   %1:3 = hlfir.associate %0 {adapt.valuebyref} :
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
      // FIXME: note that the AssociateOp that is being erased
      // here will continue to be a user of the original Source
      // operand (e.g. a result of hlfir.elemental), because
      // the erasure is not immediate in the rewriter.
      // In case there are multiple uses of the Source operand,
      // the allOtherUsesAreSafeForAssociate() below will always
      // see them, so there is no way to reuse the buffer.
      // I think we have to run this analysis before doing
      // the conversions, so that we can analyze HLFIR in its
      // original form and decide which of the AssociateOp
      // users of hlfir.expr can reuse the buffer (if it can).
      rewriter.eraseOp(associate);
    };

    // If this is the last use of the expression value and this is an hlfir.expr
    // that was bufferized, re-use the storage.
    // Otherwise, create a temp and assign the storage to it.
    //
    // WARNING: it is important to use the original Source operand
    // of the AssociateOp to look for the users, because its replacement
    // has zero materialized users at this point.
    // So allOtherUsesAreSafeForAssociate() may incorrectly return
    // true here.
    if (!isTrivialValue && allOtherUsesAreSafeForAssociate(
                               associate.getSource(), associate.getOperation(),
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
      llvm::SmallVector<mlir::NamedAttribute, 1> attrs;
      if (associate->hasAttr(fir::getAdaptToByRefAttrName())) {
        attrs.push_back(fir::getAdaptToByRefAttr(builder));
      }
      llvm::StringRef name = "";
      if (associate.getUniqName())
        name = *associate.getUniqName();
      auto temp =
          builder.createTemporary(loc, bufferizedExpr.getType(), name, attrs);
      builder.create<fir::StoreOp>(loc, bufferizedExpr, temp);
      mlir::Value mustFree = builder.createBool(loc, false);
      replaceWith(temp, temp, mustFree);
      return mlir::success();
    }
    // non-trivial value with more than one use. We will have to make a copy and
    // use that
    hlfir::Entity source = hlfir::Entity{bufferizedExpr};
    auto [temp, cleanup] = hlfir::createTempFromMold(loc, builder, source);
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

static void genBufferDestruction(mlir::Location loc, fir::FirOpBuilder &builder,
                                 mlir::Value var, mlir::Value mustFree,
                                 bool mustFinalize) {
  auto genFreeOrFinalize = [&](bool doFree, bool deallocComponents,
                               bool doFinalize) {
    if (!doFree && !deallocComponents && !doFinalize)
      return;

    mlir::Value addr = var;

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
      addr = builder.create<fir::BoxAddrOp>(loc, heapType, var);
      // Lowering currently does not produce DestroyOp with 'finalize'
      // for polymorphic temporaries. It will have to do so, for example,
      // for MERGE with polymorphic results.
      if (mustFinalize)
        TODO(loc, "finalizing polymorphic temporary in HLFIR");
    } else if (var.getType().isa<fir::BaseBoxType, fir::BoxCharType>()) {
      if (mustFinalize && !mlir::isa<fir::BaseBoxType>(var.getType()))
        fir::emitFatalError(loc, "non-finalizable variable");

      addr = builder.create<fir::BoxAddrOp>(loc, heapType, var);
    } else {
      if (!var.getType().isa<fir::HeapType>())
        addr = builder.create<fir::ConvertOp>(loc, heapType, var);

      if (mustFinalize || deallocComponents) {
        // Embox the raw pointer using proper shape and type params
        // (note that the shape might be visible via the array finalization
        // routines).
        if (!hlfir::isFortranEntity(var))
          TODO(loc, "need a Fortran entity to create a box");

        hlfir::Entity entity{var};
        llvm::SmallVector<mlir::Value> lenParams;
        hlfir::genLengthParameters(loc, builder, entity, lenParams);
        mlir::Value shape;
        if (entity.isArray())
          shape = hlfir::genShape(loc, builder, entity);
        mlir::Type boxType = fir::BoxType::get(heapType);
        var = builder.createBox(loc, boxType, addr, shape, /*slice=*/nullptr,
                                lenParams, /*tdesc=*/nullptr);
      }
    }

    if (mustFinalize)
      fir::runtime::genDerivedTypeFinalize(builder, loc, var);

    // If there are allocatable components, they need to be deallocated
    // (regardless of the mustFree and mustFinalize settings).
    if (deallocComponents)
      fir::runtime::genDerivedTypeDestroyWithoutFinalization(builder, loc, var);

    if (doFree)
      builder.create<fir::FreeMemOp>(loc, addr);
  };
  bool deallocComponents = hlfir::mayHaveAllocatableComponent(var.getType());

  auto genFree = [&]() {
    genFreeOrFinalize(/*doFree=*/true, /*deallocComponents=*/false,
                      /*doFinalize=*/false);
  };
  if (auto cstMustFree = fir::getIntIfConstant(mustFree)) {
    genFreeOrFinalize(*cstMustFree != 0 ? true : false, deallocComponents,
                      mustFinalize);
    return;
  }

  // If mustFree is dynamic, first, deallocate any allocatable
  // components and finalize.
  genFreeOrFinalize(/*doFree=*/false, deallocComponents,
                    /*doFinalize=*/mustFinalize);
  // Conditionally free the memory.
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
    genBufferDestruction(loc, builder, adaptor.getVar(), adaptor.getMustFree(),
                         /*mustFinalize=*/false);
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
      // Passing FIR base might be enough for cases when
      // component deallocation and finalization are not required.
      // If extra BoxAddr operations become a performance problem,
      // we may pass both bases and let genBufferDestruction decide
      // which one to use.
      mlir::Value base = bufferizedExpr.getBase();
      genBufferDestruction(loc, builder, base, mustFree,
                           destroy.mustFinalizeExpr());
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
    std::optional<hlfir::Entity> mold;
    if (adaptor.getMold())
      mold = getBufferizedExprStorage(adaptor.getMold());
    auto extents = hlfir::getIndexExtents(loc, builder, shape);
    auto [temp, cleanup] =
        createArrayTemp(loc, builder, elemental.getType(), shape, extents,
                        adaptor.getTypeparams(), mold);
    // If the box load is needed, we'd better place it outside
    // of the loop nest.
    temp = derefPointersAndAllocatables(loc, builder, temp);

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
    // If the elemental result is a temporary of a derived type,
    // we can avoid the deep copy implied by the AssignOp and just
    // do the shallow copy with load/store. This helps avoiding the overhead
    // of deallocating allocatable components of the temporary (if any)
    // on each iteration of the elemental operation.
    auto asExpr = elementValue.getDefiningOp<hlfir::AsExprOp>();
    auto elemType = hlfir::getFortranElementType(elementValue.getType());
    if (asExpr && asExpr.isMove() && mlir::isa<fir::RecordType>(elemType) &&
        hlfir::mayHaveAllocatableComponent(elemType) &&
        wasCreatedInCurrentBlock(elementValue, builder)) {
      auto load = builder.create<fir::LoadOp>(loc, asExpr.getVar());
      builder.create<fir::StoreOp>(loc, load, tempElement);
    } else {
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
    }
    builder.restoreInsertionPoint(insPt);

    mlir::Value bufferizedExpr =
        packageBufferizedExpr(loc, builder, temp, cleanup);
    // Explicitly delete the body of the elemental to get rid
    // of any users of hlfir.expr values inside the body as early
    // as possible.
    rewriter.startRootUpdate(elemental);
    rewriter.eraseBlock(elemental.getBody());
    rewriter.finalizeRootUpdate(elemental);
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
