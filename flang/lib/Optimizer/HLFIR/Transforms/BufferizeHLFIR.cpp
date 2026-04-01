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
#include "flang/Optimizer/OpenMP/Passes.h"
#include "aiir/Dialect/OpenMP/OpenMPDialect.h"
#include "aiir/IR/Dominance.h"
#include "aiir/IR/PatternMatch.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Pass/PassManager.h"
#include "aiir/Transforms/DialectConversion.h"
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
static aiir::Value packageBufferizedExpr(aiir::Location loc,
                                         fir::FirOpBuilder &builder,
                                         hlfir::Entity storage,
                                         aiir::Value mustFree) {
  auto tupleType = aiir::TupleType::get(
      builder.getContext(),
      aiir::TypeRange{storage.getType(), mustFree.getType()});
  auto undef = fir::UndefOp::create(builder, loc, tupleType);
  auto insert = fir::InsertValueOp::create(
      builder, loc, tupleType, undef, mustFree,
      builder.getArrayAttr(
          {builder.getIntegerAttr(builder.getIndexType(), 1)}));
  return fir::InsertValueOp::create(
      builder, loc, tupleType, insert, storage,
      builder.getArrayAttr(
          {builder.getIntegerAttr(builder.getIndexType(), 0)}));
}

/// Helper to create tuple from a bufferized expr storage and constant
/// boolean clean-up flag.
static aiir::Value packageBufferizedExpr(aiir::Location loc,
                                         fir::FirOpBuilder &builder,
                                         hlfir::Entity storage, bool mustFree) {
  aiir::Value mustFreeValue = builder.createBool(loc, mustFree);
  return packageBufferizedExpr(loc, builder, storage, mustFreeValue);
}

/// Helper to extract the storage from a tuple created by packageBufferizedExpr.
/// It assumes no tuples are used as HLFIR operation operands, which is
/// currently enforced by the verifiers that only accept HLFIR value or
/// variable types which do not include tuples.
static hlfir::Entity getBufferizedExprStorage(aiir::Value bufferizedExpr) {
  auto tupleType = aiir::dyn_cast<aiir::TupleType>(bufferizedExpr.getType());
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
static aiir::Value getBufferizedExprMustFreeFlag(aiir::Value bufferizedExpr) {
  auto tupleType = aiir::dyn_cast<aiir::TupleType>(bufferizedExpr.getType());
  if (!tupleType)
    return bufferizedExpr;
  assert(tupleType.size() == 2 && "unexpected tuple type");
  if (auto insert = bufferizedExpr.getDefiningOp<fir::InsertValueOp>())
    if (auto insert0 = insert.getAdt().getDefiningOp<fir::InsertValueOp>())
      if (insert0.getVal().getType() == tupleType.getType(1))
        return insert0.getVal();
  TODO(bufferizedExpr.getLoc(), "general extract storage case");
}

static std::pair<hlfir::Entity, aiir::Value>
createArrayTemp(aiir::Location loc, fir::FirOpBuilder &builder,
                aiir::Type exprType, aiir::Value shape,
                llvm::ArrayRef<aiir::Value> extents,
                llvm::ArrayRef<aiir::Value> lenParams,
                std::optional<hlfir::Entity> polymorphicMold) {
  auto sequenceType = aiir::cast<fir::SequenceType>(
      hlfir::getFortranElementOrSequenceType(exprType));

  auto genTempDeclareOp =
      [](fir::FirOpBuilder &builder, aiir::Location loc, aiir::Value memref,
         llvm::StringRef name, aiir::Value shape,
         llvm::ArrayRef<aiir::Value> typeParams,
         fir::FortranVariableFlagsAttr attrs) -> aiir::Value {
    auto declareOp =
        hlfir::DeclareOp::create(builder, loc, memref, name, shape, typeParams,
                                 /*dummy_scope=*/nullptr, /*storage=*/nullptr,
                                 /*storage_offset=*/0, attrs);
    return declareOp.getBase();
  };

  auto [base, isHeapAlloc] = builder.createArrayTemp(
      loc, sequenceType, shape, extents, lenParams, genTempDeclareOp,
      polymorphicMold ? polymorphicMold->getFirBase() : nullptr);
  hlfir::Entity temp = hlfir::Entity{base};
  assert(!temp.isAllocatable() && "temp must have been allocated");
  return {temp, builder.createBool(loc, isHeapAlloc)};
}

/// Copy \p source into a new temporary and package the temporary into a
/// <temp,cleanup> tuple. The temporary may be heap or stack allocated.
static aiir::Value copyInTempAndPackage(aiir::Location loc,
                                        fir::FirOpBuilder &builder,
                                        hlfir::Entity source) {
  auto [temp, mustFree] = hlfir::createTempFromMold(loc, builder, source);
  assert(!temp.isAllocatable() && "expect temp to already be allocated");
  hlfir::AssignOp::create(builder, loc, source, temp, /*realloc=*/false,
                          /*keep_lhs_length_if_realloc=*/false,
                          /*temporary_lhs=*/true);
  return packageBufferizedExpr(loc, builder, temp, mustFree);
}

struct AsExprOpConversion : public aiir::OpConversionPattern<hlfir::AsExprOp> {
  using aiir::OpConversionPattern<hlfir::AsExprOp>::OpConversionPattern;
  explicit AsExprOpConversion(aiir::AIIRContext *ctx)
      : aiir::OpConversionPattern<hlfir::AsExprOp>{ctx} {}
  llvm::LogicalResult
  matchAndRewrite(hlfir::AsExprOp asExpr, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::Location loc = asExpr->getLoc();
    auto module = asExpr->getParentOfType<aiir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, module);
    if (asExpr.isMove()) {
      // Move variable storage for the hlfir.expr buffer.
      aiir::Value bufferizedExpr = packageBufferizedExpr(
          loc, builder, hlfir::Entity{adaptor.getVar()}, adaptor.getMustFree());
      rewriter.replaceOp(asExpr, bufferizedExpr);
      return aiir::success();
    }
    // Otherwise, create a copy in a new buffer.
    hlfir::Entity source = hlfir::Entity{adaptor.getVar()};
    aiir::Value bufferizedExpr = copyInTempAndPackage(loc, builder, source);
    rewriter.replaceOp(asExpr, bufferizedExpr);
    return aiir::success();
  }
};

struct ShapeOfOpConversion
    : public aiir::OpConversionPattern<hlfir::ShapeOfOp> {
  using aiir::OpConversionPattern<hlfir::ShapeOfOp>::OpConversionPattern;

  llvm::LogicalResult
  matchAndRewrite(hlfir::ShapeOfOp shapeOf, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::Location loc = shapeOf.getLoc();
    aiir::ModuleOp mod = shapeOf->getParentOfType<aiir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);

    aiir::Value shape;
    hlfir::Entity bufferizedExpr{getBufferizedExprStorage(adaptor.getExpr())};
    if (bufferizedExpr.isVariable()) {
      shape = hlfir::genShape(loc, builder, bufferizedExpr);
    } else {
      // everything else failed so try to create a shape from static type info
      hlfir::ExprType exprTy =
          aiir::dyn_cast_or_null<hlfir::ExprType>(adaptor.getExpr().getType());
      if (exprTy)
        shape = hlfir::genExprShape(builder, loc, exprTy);
    }
    // expected to never happen
    if (!shape)
      return emitError(loc,
                       "Unresolvable hlfir.shape_of where extents are unknown");

    rewriter.replaceOp(shapeOf, shape);
    return aiir::success();
  }
};

struct ApplyOpConversion : public aiir::OpConversionPattern<hlfir::ApplyOp> {
  using aiir::OpConversionPattern<hlfir::ApplyOp>::OpConversionPattern;
  explicit ApplyOpConversion(aiir::AIIRContext *ctx)
      : aiir::OpConversionPattern<hlfir::ApplyOp>{ctx} {}
  llvm::LogicalResult
  matchAndRewrite(hlfir::ApplyOp apply, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::Location loc = apply->getLoc();
    hlfir::Entity bufferizedExpr = getBufferizedExprStorage(adaptor.getExpr());
    aiir::Type resultType = hlfir::getVariableElementType(bufferizedExpr);
    aiir::Value result = hlfir::DesignateOp::create(
        rewriter, loc, resultType, bufferizedExpr, adaptor.getIndices(),
        adaptor.getTypeparams());
    if (fir::isa_trivial(apply.getType())) {
      result = fir::LoadOp::create(rewriter, loc, result);
    } else {
      fir::FirOpBuilder builder(rewriter, apply.getOperation());
      result =
          packageBufferizedExpr(loc, builder, hlfir::Entity{result}, false);
    }
    rewriter.replaceOp(apply, result);
    return aiir::success();
  }
};

struct AssignOpConversion : public aiir::OpConversionPattern<hlfir::AssignOp> {
  using aiir::OpConversionPattern<hlfir::AssignOp>::OpConversionPattern;
  explicit AssignOpConversion(aiir::AIIRContext *ctx)
      : aiir::OpConversionPattern<hlfir::AssignOp>{ctx} {}
  llvm::LogicalResult
  matchAndRewrite(hlfir::AssignOp assign, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<aiir::Value> newOperands;
    for (aiir::Value operand : adaptor.getOperands())
      newOperands.push_back(getBufferizedExprStorage(operand));
    rewriter.startOpModification(assign);
    assign->setOperands(newOperands);
    rewriter.finalizeOpModification(assign);
    return aiir::success();
  }
};

struct ConcatOpConversion : public aiir::OpConversionPattern<hlfir::ConcatOp> {
  using aiir::OpConversionPattern<hlfir::ConcatOp>::OpConversionPattern;
  explicit ConcatOpConversion(aiir::AIIRContext *ctx)
      : aiir::OpConversionPattern<hlfir::ConcatOp>{ctx} {}
  llvm::LogicalResult
  matchAndRewrite(hlfir::ConcatOp concat, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::Location loc = concat->getLoc();
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
    aiir::Type addrType = fir::ReferenceType::get(
        hlfir::getFortranElementType(concat.getResult().getType()));
    aiir::Value cast = builder.createConvert(loc, addrType, fir::getBase(res));
    res = fir::substBase(res, cast);
    hlfir::Entity hlfirTempRes =
        hlfir::Entity{hlfir::genDeclare(loc, builder, res, "tmp",
                                        fir::FortranVariableFlagsAttr{})
                          .getBase()};
    aiir::Value bufferizedExpr =
        packageBufferizedExpr(loc, builder, hlfirTempRes, false);
    rewriter.replaceOp(concat, bufferizedExpr);
    return aiir::success();
  }
};

struct SetLengthOpConversion
    : public aiir::OpConversionPattern<hlfir::SetLengthOp> {
  using aiir::OpConversionPattern<hlfir::SetLengthOp>::OpConversionPattern;
  explicit SetLengthOpConversion(aiir::AIIRContext *ctx)
      : aiir::OpConversionPattern<hlfir::SetLengthOp>{ctx} {}
  llvm::LogicalResult
  matchAndRewrite(hlfir::SetLengthOp setLength, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::Location loc = setLength->getLoc();
    fir::FirOpBuilder builder(rewriter, setLength.getOperation());
    // Create a temp with the new length.
    hlfir::Entity string = getBufferizedExprStorage(adaptor.getString());
    auto charType = hlfir::getFortranElementType(setLength.getType());
    llvm::StringRef tmpName{".tmp"};
    llvm::SmallVector<aiir::Value, 1> lenParams{adaptor.getLength()};
    auto alloca = builder.createTemporary(loc, charType, tmpName,
                                          /*shape=*/{}, lenParams);
    auto declareOp = hlfir::DeclareOp::create(
        builder, loc, alloca, tmpName, /*shape=*/aiir::Value{}, lenParams);
    hlfir::Entity temp{declareOp.getBase()};
    // Assign string value to the created temp.
    hlfir::AssignOp::create(builder, loc, string, temp,
                            /*realloc=*/false,
                            /*keep_lhs_length_if_realloc=*/false,
                            /*temporary_lhs=*/true);
    aiir::Value bufferizedExpr =
        packageBufferizedExpr(loc, builder, temp, false);
    rewriter.replaceOp(setLength, bufferizedExpr);
    return aiir::success();
  }
};

struct GetLengthOpConversion
    : public aiir::OpConversionPattern<hlfir::GetLengthOp> {
  using aiir::OpConversionPattern<hlfir::GetLengthOp>::OpConversionPattern;
  explicit GetLengthOpConversion(aiir::AIIRContext *ctx)
      : aiir::OpConversionPattern<hlfir::GetLengthOp>{ctx} {}
  llvm::LogicalResult
  matchAndRewrite(hlfir::GetLengthOp getLength, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::Location loc = getLength->getLoc();
    fir::FirOpBuilder builder(rewriter, getLength.getOperation());
    hlfir::Entity bufferizedExpr = getBufferizedExprStorage(adaptor.getExpr());
    aiir::Value length = hlfir::genCharLength(loc, builder, bufferizedExpr);
    if (!length)
      return rewriter.notifyMatchFailure(
          getLength, "could not deduce length from GetLengthOp operand");
    length = builder.createConvert(loc, builder.getIndexType(), length);
    rewriter.replaceOp(getLength, length);
    return aiir::success();
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
static bool allOtherUsesAreSafeForAssociate(aiir::Value value,
                                            aiir::Operation *currentUse,
                                            aiir::Operation *endAssociate) {
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

  for (aiir::Operation *useOp : value.getUsers()) {
    // Ignore DestroyOp's that do not imply finalization.
    // If finalization is implied, then we must delegate
    // the finalization to the correspoding EndAssociateOp,
    // but we currently do not; so we disable the buffer
    // reuse in this case.
    if (auto destroy = aiir::dyn_cast<hlfir::DestroyOp>(useOp)) {
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
      if (aiir::isa<hlfir::ShapeOfOp>(useOp) ||
          aiir::isa<hlfir::GetLengthOp>(useOp)) {
        if (!endAssociate)
          continue;
        // If useOp dominates the endAssociate, then it is definitely safe.
        if (useOp->getBlock() != endAssociate->getBlock())
          if (aiir::DominanceInfo{}.dominates(useOp, endAssociate))
            continue;
        if (useOp->isBeforeInBlock(endAssociate))
          continue;
      }
      return false;
    }
  }
  return true;
}

static void eraseAllUsesInDestroys(aiir::Value value,
                                   aiir::ConversionPatternRewriter &rewriter) {
  for (aiir::Operation *useOp : value.getUsers())
    if (auto destroy = aiir::dyn_cast<hlfir::DestroyOp>(useOp)) {
      assert(!destroy.mustFinalizeExpr() &&
             "deleting DestroyOp with finalize attribute");
      rewriter.eraseOp(destroy);
    }
}

struct AssociateOpConversion
    : public aiir::OpConversionPattern<hlfir::AssociateOp> {
  using aiir::OpConversionPattern<hlfir::AssociateOp>::OpConversionPattern;
  explicit AssociateOpConversion(aiir::AIIRContext *ctx)
      : aiir::OpConversionPattern<hlfir::AssociateOp>{ctx} {}
  llvm::LogicalResult
  matchAndRewrite(hlfir::AssociateOp associate, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::Location loc = associate->getLoc();
    fir::FirOpBuilder builder(rewriter, associate.getOperation());
    aiir::Value bufferizedExpr = getBufferizedExprStorage(adaptor.getSource());
    const bool isTrivialValue = fir::isa_trivial(bufferizedExpr.getType());

    auto getEndAssociate =
        [](hlfir::AssociateOp associate) -> aiir::Operation * {
      for (aiir::Operation *useOp : associate->getUsers())
        if (aiir::isa<hlfir::EndAssociateOp>(useOp))
          return useOp;
      // happens in some hand coded aiir in tests
      return nullptr;
    };

    auto replaceWith = [&](aiir::Value hlfirVar, aiir::Value firVar,
                           aiir::Value flag) {
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
      auto adjustVar = [&](aiir::Value sourceVar, aiir::Type assocType) {
        if ((aiir::isa<fir::BaseBoxType>(sourceVar.getType()) &&
             !aiir::isa<fir::BaseBoxType>(assocType)) ||
            ((aiir::isa<fir::BoxCharType>(sourceVar.getType()) &&
              !aiir::isa<fir::BoxCharType>(assocType)))) {
          sourceVar =
              fir::BoxAddrOp::create(builder, loc, assocType, sourceVar);
        } else {
          sourceVar = builder.createConvert(loc, assocType, sourceVar);
        }
        return sourceVar;
      };

      aiir::Type associateHlfirVarType = associate.getResultTypes()[0];
      hlfirVar = adjustVar(hlfirVar, associateHlfirVarType);
      aiir::Type associateFirVarType = associate.getResultTypes()[1];
      firVar = adjustVar(firVar, associateFirVarType);
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
      rewriter.replaceOp(associate, {hlfirVar, firVar, flag});
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
      aiir::Value mustFree = getBufferizedExprMustFreeFlag(adaptor.getSource());
      aiir::Value firBase = hlfir::Entity{bufferizedExpr}.getFirBase();
      replaceWith(bufferizedExpr, firBase, mustFree);
      eraseAllUsesInDestroys(associate.getSource(), rewriter);
      // Make sure to erase the hlfir.destroy if there is an indirection through
      // a hlfir.no_reassoc operation.
      if (auto noReassoc = aiir::dyn_cast_or_null<hlfir::NoReassocOp>(
              associate.getSource().getDefiningOp()))
        eraseAllUsesInDestroys(noReassoc.getVal(), rewriter);
      return aiir::success();
    }
    if (isTrivialValue) {
      llvm::SmallVector<aiir::NamedAttribute, 1> attrs;
      if (associate->hasAttr(fir::getAdaptToByRefAttrName())) {
        attrs.push_back(fir::getAdaptToByRefAttr(builder));
      }
      llvm::StringRef name = "";
      if (associate.getUniqName())
        name = *associate.getUniqName();
      auto temp =
          builder.createTemporary(loc, bufferizedExpr.getType(), name, attrs);
      fir::StoreOp::create(builder, loc, bufferizedExpr, temp);
      aiir::Value mustFree = builder.createBool(loc, false);
      replaceWith(temp, temp, mustFree);
      return aiir::success();
    }
    // non-trivial value with more than one use. We will have to make a copy and
    // use that
    hlfir::Entity source = hlfir::Entity{bufferizedExpr};
    aiir::Value bufferTuple = copyInTempAndPackage(loc, builder, source);
    bufferizedExpr = getBufferizedExprStorage(bufferTuple);
    replaceWith(bufferizedExpr, hlfir::Entity{bufferizedExpr}.getFirBase(),
                getBufferizedExprMustFreeFlag(bufferTuple));
    return aiir::success();
  }
};

static void genBufferDestruction(aiir::Location loc, fir::FirOpBuilder &builder,
                                 aiir::Value var, aiir::Value mustFree,
                                 bool mustFinalize) {
  auto genFreeOrFinalize = [&](bool doFree, bool deallocComponents,
                               bool doFinalize) {
    if (!doFree && !deallocComponents && !doFinalize)
      return;

    aiir::Value addr = var;

    // fir::FreeMemOp operand type must be a fir::HeapType.
    aiir::Type heapType = fir::HeapType::get(
        hlfir::getFortranElementOrSequenceType(var.getType()));
    if (aiir::isa<fir::BaseBoxType, fir::BoxCharType>(var.getType())) {
      if (mustFinalize && !aiir::isa<fir::BaseBoxType>(var.getType()))
        fir::emitFatalError(loc, "non-finalizable variable");

      addr = fir::BoxAddrOp::create(builder, loc, heapType, var);
    } else {
      if (!aiir::isa<fir::HeapType>(var.getType()))
        addr = fir::ConvertOp::create(builder, loc, heapType, var);

      if (mustFinalize || deallocComponents) {
        // Embox the raw pointer using proper shape and type params
        // (note that the shape might be visible via the array finalization
        // routines).
        if (!hlfir::isFortranEntity(var))
          TODO(loc, "need a Fortran entity to create a box");

        hlfir::Entity entity{var};
        llvm::SmallVector<aiir::Value> lenParams;
        hlfir::genLengthParameters(loc, builder, entity, lenParams);
        aiir::Value shape;
        if (entity.isArray())
          shape = hlfir::genShape(loc, builder, entity);
        aiir::Type boxType = fir::BoxType::get(heapType);
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
      fir::FreeMemOp::create(builder, loc, addr);
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
    : public aiir::OpConversionPattern<hlfir::EndAssociateOp> {
  using aiir::OpConversionPattern<hlfir::EndAssociateOp>::OpConversionPattern;
  explicit EndAssociateOpConversion(aiir::AIIRContext *ctx)
      : aiir::OpConversionPattern<hlfir::EndAssociateOp>{ctx} {}
  llvm::LogicalResult
  matchAndRewrite(hlfir::EndAssociateOp endAssociate, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::Location loc = endAssociate->getLoc();
    fir::FirOpBuilder builder(rewriter, endAssociate.getOperation());
    genBufferDestruction(loc, builder, adaptor.getVar(), adaptor.getMustFree(),
                         /*mustFinalize=*/false);
    rewriter.eraseOp(endAssociate);
    return aiir::success();
  }
};

struct DestroyOpConversion
    : public aiir::OpConversionPattern<hlfir::DestroyOp> {
  using aiir::OpConversionPattern<hlfir::DestroyOp>::OpConversionPattern;
  explicit DestroyOpConversion(aiir::AIIRContext *ctx)
      : aiir::OpConversionPattern<hlfir::DestroyOp>{ctx} {}
  llvm::LogicalResult
  matchAndRewrite(hlfir::DestroyOp destroy, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    // If expr was bufferized on the heap, now is time to deallocate the buffer.
    aiir::Location loc = destroy->getLoc();
    hlfir::Entity bufferizedExpr = getBufferizedExprStorage(adaptor.getExpr());
    if (!fir::isa_trivial(bufferizedExpr.getType())) {
      fir::FirOpBuilder builder(rewriter, destroy.getOperation());
      aiir::Value mustFree = getBufferizedExprMustFreeFlag(adaptor.getExpr());
      // Passing FIR base might be enough for cases when
      // component deallocation and finalization are not required.
      // If extra BoxAddr operations become a performance problem,
      // we may pass both bases and let genBufferDestruction decide
      // which one to use.
      aiir::Value base = bufferizedExpr.getBase();
      genBufferDestruction(loc, builder, base, mustFree,
                           destroy.mustFinalizeExpr());
    }

    rewriter.eraseOp(destroy);
    return aiir::success();
  }
};

struct NoReassocOpConversion
    : public aiir::OpConversionPattern<hlfir::NoReassocOp> {
  using aiir::OpConversionPattern<hlfir::NoReassocOp>::OpConversionPattern;
  explicit NoReassocOpConversion(aiir::AIIRContext *ctx)
      : aiir::OpConversionPattern<hlfir::NoReassocOp>{ctx} {}
  llvm::LogicalResult
  matchAndRewrite(hlfir::NoReassocOp noreassoc, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::Location loc = noreassoc->getLoc();
    fir::FirOpBuilder builder(rewriter, noreassoc.getOperation());
    aiir::Value bufferizedExpr = getBufferizedExprStorage(adaptor.getVal());
    aiir::Value result =
        hlfir::NoReassocOp::create(builder, loc, bufferizedExpr);

    if (!fir::isa_trivial(bufferizedExpr.getType())) {
      // NoReassocOp should not be needed on the mustFree path.
      aiir::Value mustFree = getBufferizedExprMustFreeFlag(adaptor.getVal());
      result =
          packageBufferizedExpr(loc, builder, hlfir::Entity{result}, mustFree);
    }
    rewriter.replaceOp(noreassoc, result);
    return aiir::success();
  }
};

/// Was \p value created in the aiir block where \p builder is currently set ?
static bool wasCreatedInCurrentBlock(aiir::Value value,
                                     fir::FirOpBuilder &builder) {
  if (aiir::Operation *op = value.getDefiningOp())
    return op->getBlock() == builder.getBlock();
  return false;
}

/// This Listener allows setting both the builder and the rewriter as
/// listeners. This is required when a pattern uses a firBuilder helper that
/// may create illegal operations that will need to be translated and requires
/// notifying the rewriter.
struct HLFIRListener : public aiir::OpBuilder::Listener {
  HLFIRListener(fir::FirOpBuilder &builder,
                aiir::ConversionPatternRewriter &rewriter)
      : builder{builder}, rewriter{rewriter} {}
  void notifyOperationInserted(aiir::Operation *op,
                               aiir::OpBuilder::InsertPoint previous) override {
    builder.notifyOperationInserted(op, previous);
    rewriter.getListener()->notifyOperationInserted(op, previous);
  }
  virtual void notifyBlockInserted(aiir::Block *block, aiir::Region *previous,
                                   aiir::Region::iterator previousIt) override {
    builder.notifyBlockInserted(block, previous, previousIt);
    rewriter.getListener()->notifyBlockInserted(block, previous, previousIt);
  }
  fir::FirOpBuilder &builder;
  aiir::ConversionPatternRewriter &rewriter;
};

struct ElementalOpConversion
    : public aiir::OpConversionPattern<hlfir::ElementalOp> {
  using aiir::OpConversionPattern<hlfir::ElementalOp>::OpConversionPattern;
  explicit ElementalOpConversion(aiir::AIIRContext *ctx,
                                 bool optimizeEmptyElementals = false)
      : aiir::OpConversionPattern<hlfir::ElementalOp>{ctx},
        optimizeEmptyElementals(optimizeEmptyElementals) {
    // This pattern recursively converts nested ElementalOp's
    // by cloning and then converting them, so we have to allow
    // for recursive pattern application. The recursion is bounded
    // by the nesting level of ElementalOp's.
    setHasBoundedRewriteRecursion();
  }
  llvm::LogicalResult
  matchAndRewrite(hlfir::ElementalOp elemental, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::Location loc = elemental->getLoc();
    fir::FirOpBuilder builder(rewriter, elemental.getOperation());
    // The body of the elemental op may contain operation that will require
    // to be translated. Notify the rewriter about the cloned operations.
    HLFIRListener listener{builder, rewriter};
    builder.setListener(&listener);

    aiir::Value shape = adaptor.getShape();
    std::optional<hlfir::Entity> mold;
    if (adaptor.getMold())
      mold = getBufferizedExprStorage(adaptor.getMold());
    auto extents = hlfir::getIndexExtents(loc, builder, shape);
    llvm::SmallVector<aiir::Value> typeParams(adaptor.getTypeparams().begin(),
                                              adaptor.getTypeparams().end());
    auto [temp, cleanup] = createArrayTemp(loc, builder, elemental.getType(),
                                           shape, extents, typeParams, mold);

    if (optimizeEmptyElementals)
      extents = fir::factory::updateRuntimeExtentsForEmptyArrays(builder, loc,
                                                                 extents);

    // Generate a loop nest looping around the fir.elemental shape and clone
    // fir.elemental region inside the inner loop.
    hlfir::LoopNest loopNest =
        hlfir::genLoopNest(loc, builder, extents, !elemental.isOrdered(),
                           flangomp::shouldUseWorkshareLowering(elemental));
    auto insPt = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(loopNest.body);
    auto yield = hlfir::inlineElementalOp(loc, builder, elemental,
                                          loopNest.oneBasedIndices);
    hlfir::Entity elementValue(yield.getElementValue());
    // Skip final AsExpr if any. It would create an element temporary,
    // which is no needed since the element will be assigned right away in
    // the array temporary. An hlfir.as_expr may have been added if the
    // elemental is a "view" over a variable (e.g parentheses or transpose).
    if (auto asExpr = elementValue.getDefiningOp<hlfir::AsExprOp>()) {
      if (asExpr->hasOneUse() && !asExpr.isMove()) {
        // Check that the asExpr is the final operation before the yield,
        // otherwise, clean-ups could impact the memory being re-used.
        if (asExpr->getNextNode() == yield.getOperation()) {
          elementValue = hlfir::Entity{asExpr.getVar()};
          rewriter.eraseOp(asExpr);
        }
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
    if (asExpr && asExpr.isMove() && aiir::isa<fir::RecordType>(elemType) &&
        hlfir::mayHaveAllocatableComponent(elemType) &&
        wasCreatedInCurrentBlock(elementValue, builder)) {
      auto load = fir::LoadOp::create(builder, loc, asExpr.getVar());
      fir::StoreOp::create(builder, loc, load, tempElement);
    } else {
      hlfir::AssignOp::create(builder, loc, elementValue, tempElement,
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
      if (aiir::isa<hlfir::ExprType>(elementValue.getType()) &&
          wasCreatedInCurrentBlock(elementValue, builder))
        hlfir::DestroyOp::create(builder, loc, elementValue);
    }
    builder.restoreInsertionPoint(insPt);

    aiir::Value bufferizedExpr =
        packageBufferizedExpr(loc, builder, temp, cleanup);
    // Explicitly delete the body of the elemental to get rid
    // of any users of hlfir.expr values inside the body as early
    // as possible.
    rewriter.startOpModification(elemental);
    rewriter.eraseBlock(elemental.getBody());
    rewriter.finalizeOpModification(elemental);
    rewriter.replaceOp(elemental, bufferizedExpr);
    return aiir::success();
  }

private:
  bool optimizeEmptyElementals = false;
};
struct CharExtremumOpConversion
    : public aiir::OpConversionPattern<hlfir::CharExtremumOp> {
  using aiir::OpConversionPattern<hlfir::CharExtremumOp>::OpConversionPattern;
  explicit CharExtremumOpConversion(aiir::AIIRContext *ctx)
      : aiir::OpConversionPattern<hlfir::CharExtremumOp>{ctx} {}
  llvm::LogicalResult
  matchAndRewrite(hlfir::CharExtremumOp char_extremum, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::Location loc = char_extremum->getLoc();
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
    aiir::Type addrType = fir::ReferenceType::get(
        hlfir::getFortranElementType(char_extremum.getResult().getType()));
    aiir::Value cast = builder.createConvert(loc, addrType, fir::getBase(res));
    res = fir::substBase(res, cast);
    hlfir::Entity hlfirTempRes =
        hlfir::Entity{hlfir::genDeclare(loc, builder, res, ".tmp.char_extremum",
                                        fir::FortranVariableFlagsAttr{})
                          .getBase()};
    aiir::Value bufferizedExpr =
        packageBufferizedExpr(loc, builder, hlfirTempRes, false);
    rewriter.replaceOp(char_extremum, bufferizedExpr);
    return aiir::success();
  }
};

struct EvaluateInMemoryOpConversion
    : public aiir::OpConversionPattern<hlfir::EvaluateInMemoryOp> {
  using aiir::OpConversionPattern<
      hlfir::EvaluateInMemoryOp>::OpConversionPattern;
  explicit EvaluateInMemoryOpConversion(aiir::AIIRContext *ctx)
      : aiir::OpConversionPattern<hlfir::EvaluateInMemoryOp>{ctx} {}
  llvm::LogicalResult
  matchAndRewrite(hlfir::EvaluateInMemoryOp evalInMemOp, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::Location loc = evalInMemOp->getLoc();
    fir::FirOpBuilder builder(rewriter, evalInMemOp.getOperation());
    auto [temp, isHeapAlloc] = hlfir::computeEvaluateOpInNewTemp(
        loc, builder, evalInMemOp, adaptor.getShape(), adaptor.getTypeparams());
    aiir::Value bufferizedExpr =
        packageBufferizedExpr(loc, builder, temp, isHeapAlloc);
    rewriter.replaceOp(evalInMemOp, bufferizedExpr);
    return aiir::success();
  }
};

class BufferizeHLFIR : public hlfir::impl::BufferizeHLFIRBase<BufferizeHLFIR> {
public:
  using BufferizeHLFIRBase<BufferizeHLFIR>::BufferizeHLFIRBase;

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
    aiir::RewritePatternSet patterns(context);
    patterns.insert<ApplyOpConversion, AsExprOpConversion, AssignOpConversion,
                    AssociateOpConversion, CharExtremumOpConversion,
                    ConcatOpConversion, DestroyOpConversion,
                    EndAssociateOpConversion, EvaluateInMemoryOpConversion,
                    NoReassocOpConversion, SetLengthOpConversion,
                    ShapeOfOpConversion, GetLengthOpConversion>(context);
    patterns.insert<ElementalOpConversion>(context, optimizeEmptyElementals);
    aiir::ConversionTarget target(*context);
    // Note that YieldElementOp is not marked as an illegal operation.
    // It must be erased by its parent converter and there is no explicit
    // conversion pattern to YieldElementOp itself. If any YieldElementOp
    // survives this pass, the verifier will detect it because it has to be
    // a child of ElementalOp and ElementalOp's are explicitly illegal.
    target.addIllegalOp<hlfir::ApplyOp, hlfir::AssociateOp, hlfir::ElementalOp,
                        hlfir::EndAssociateOp, hlfir::SetLengthOp>();

    target.markUnknownOpDynamicallyLegal([](aiir::Operation *op) {
      return llvm::all_of(op->getResultTypes(),
                          [](aiir::Type ty) {
                            return !aiir::isa<hlfir::ExprType>(ty);
                          }) &&
             llvm::all_of(op->getOperandTypes(), [](aiir::Type ty) {
               return !aiir::isa<hlfir::ExprType>(ty);
             });
    });
    if (aiir::failed(
            aiir::applyFullConversion(module, target, std::move(patterns)))) {
      aiir::emitError(aiir::UnknownLoc::get(context),
                      "failure in HLFIR bufferization pass");
      signalPassFailure();
    }
  }
};
} // namespace
