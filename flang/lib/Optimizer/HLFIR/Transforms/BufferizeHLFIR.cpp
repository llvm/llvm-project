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
#include "flang/Optimizer/Builder/Runtime/Assign.h"
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
#include "mlir/Transforms/DialectConversion.h"
#include <mlir/Support/LogicalResult.h>

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
  if (mold.isPolymorphic())
    TODO(loc, "creating polymorphic temporary");
  llvm::SmallVector<mlir::Value> lenParams;
  hlfir::genLengthParameters(loc, builder, mold, lenParams);
  llvm::StringRef tmpName{".tmp"};
  mlir::Value alloc;
  mlir::Value isHeapAlloc;
  mlir::Value shape{};
  if (mold.isArray()) {
    mlir::Type sequenceType =
        hlfir::getFortranElementOrSequenceType(mold.getType());
    shape = hlfir::genShape(loc, builder, mold);
    auto extents = hlfir::getIndexExtents(loc, builder, shape);
    alloc = builder.createHeapTemporary(loc, sequenceType, tmpName, extents,
                                        lenParams);
    isHeapAlloc = builder.createBool(loc, true);
  } else {
    alloc = builder.createTemporary(loc, mold.getFortranElementType(), tmpName,
                                    /*shape*/ std::nullopt, lenParams);
    isHeapAlloc = builder.createBool(loc, false);
  }
  auto declareOp = builder.create<hlfir::DeclareOp>(
      loc, alloc, tmpName, shape, lenParams, fir::FortranVariableFlagsAttr{});
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
    fir::FirOpBuilder builder(rewriter, fir::getKindMapping(module));
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
    builder.create<hlfir::AssignOp>(loc, source, temp);
    mlir::Value bufferizedExpr =
        packageBufferizedExpr(loc, builder, temp, cleanup);
    rewriter.replaceOp(asExpr, bufferizedExpr);
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
    if (fir::isa_trivial(apply.getType()))
      result = rewriter.create<fir::LoadOp>(loc, result);
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
    rewriter.replaceOpWithNewOp<hlfir::AssignOp>(
        assign, getBufferizedExprStorage(adaptor.getOperands()[0]),
        getBufferizedExprStorage(adaptor.getOperands()[1]));
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
    auto module = concat->getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, fir::getKindMapping(module));
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
    auto module = setLength->getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, fir::getKindMapping(module));
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
    builder.create<hlfir::AssignOp>(loc, string, temp);
    mlir::Value bufferizedExpr =
        packageBufferizedExpr(loc, builder, temp, false);
    rewriter.replaceOp(setLength, bufferizedExpr);
    return mlir::success();
  }
};

static bool allOtherUsesAreDestroys(mlir::Value value,
                                    mlir::Operation *currentUse) {
  for (mlir::Operation *useOp : value.getUsers())
    if (!mlir::isa<hlfir::DestroyOp>(useOp) && useOp != currentUse)
      return false;
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
    auto module = associate->getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, fir::getKindMapping(module));
    mlir::Value bufferizedExpr = getBufferizedExprStorage(adaptor.getSource());
    const bool isTrivialValue = fir::isa_trivial(bufferizedExpr.getType());

    auto replaceWith = [&](mlir::Value hlfirVar, mlir::Value firVar,
                           mlir::Value flag) {
      hlfirVar =
          builder.createConvert(loc, associate.getResultTypes()[0], hlfirVar);
      associate.getResult(0).replaceAllUsesWith(hlfirVar);
      mlir::Type associateFirVarType = associate.getResultTypes()[1];
      if (firVar.getType().isa<fir::BaseBoxType>() &&
          !associateFirVarType.isa<fir::BaseBoxType>())
        firVar =
            builder.create<fir::BoxAddrOp>(loc, associateFirVarType, firVar);
      else
        firVar = builder.createConvert(loc, associateFirVarType, firVar);
      associate.getResult(1).replaceAllUsesWith(firVar);
      associate.getResult(2).replaceAllUsesWith(flag);
      rewriter.replaceOp(associate, {hlfirVar, firVar, flag});
    };

    // If this is the last use of the expression value and this is an hlfir.expr
    // that was bufferized, re-use the storage.
    // Otherwise, create a temp and assign the storage to it.
    if (!isTrivialValue && allOtherUsesAreDestroys(associate.getSource(),
                                                   associate.getOperation())) {
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
    TODO(loc, "hlfir.associate of hlfir.expr with more than one use");
  }
};

static void genFreeIfMustFree(mlir::Location loc, fir::FirOpBuilder &builder,
                              mlir::Value var, mlir::Value mustFree) {
  auto genFree = [&]() {
    // fir::FreeMemOp operand type must be a fir::HeapType.
    mlir::Type heapType = fir::HeapType::get(
        hlfir::getFortranElementOrSequenceType(var.getType()));
    if (var.getType().isa<fir::BaseBoxType, fir::BoxCharType>())
      var = builder.create<fir::BoxAddrOp>(loc, heapType, var);
    else if (!var.getType().isa<fir::HeapType>())
      var = builder.create<fir::ConvertOp>(loc, heapType, var);
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
    auto module = endAssociate->getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, fir::getKindMapping(module));
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
      auto module = destroy->getParentOfType<mlir::ModuleOp>();
      fir::FirOpBuilder builder(rewriter, fir::getKindMapping(module));
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
    rewriter.replaceOpWithNewOp<hlfir::NoReassocOp>(
        noreassoc, getBufferizedExprStorage(adaptor.getVal()));
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
      : mlir::OpConversionPattern<hlfir::ElementalOp>{ctx} {}
  mlir::LogicalResult
  matchAndRewrite(hlfir::ElementalOp elemental, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = elemental->getLoc();
    auto module = elemental->getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, fir::getKindMapping(module));
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
    auto [innerLoop, oneBasedLoopIndices] =
        hlfir::genLoopNest(loc, builder, extents);
    auto insPt = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(innerLoop.getBody());
    auto yield =
        hlfir::inlineElementalOp(loc, builder, elemental, oneBasedLoopIndices);
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
        hlfir::getElementAt(loc, builder, temp, oneBasedLoopIndices);
    builder.create<hlfir::AssignOp>(loc, elementValue, tempElement);
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
    patterns
        .insert<ApplyOpConversion, AsExprOpConversion, AssignOpConversion,
                AssociateOpConversion, ConcatOpConversion, DestroyOpConversion,
                ElementalOpConversion, EndAssociateOpConversion,
                NoReassocOpConversion, SetLengthOpConversion>(context);
    mlir::ConversionTarget target(*context);
    target.addIllegalOp<hlfir::ApplyOp, hlfir::AssociateOp, hlfir::ElementalOp,
                        hlfir::EndAssociateOp, hlfir::SetLengthOp,
                        hlfir::YieldElementOp>();
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
