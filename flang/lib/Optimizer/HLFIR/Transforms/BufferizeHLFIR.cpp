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
#include "flang/Optimizer/Builder/Runtime/Assign.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include <optional>

namespace hlfir {
#define GEN_PASS_DEF_BUFFERIZEHLFIR
#include "flang/Optimizer/HLFIR/Passes.h.inc"
} // namespace hlfir

namespace {

/// Helper to create tuple from a bufferized expr storage and clean up
/// instruction flag.
static mlir::Value packageBufferizedExpr(mlir::Location loc,
                                         fir::FirOpBuilder &builder,
                                         mlir::Value storage,
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
                                         mlir::Value storage, bool mustFree) {
  mlir::Value mustFreeValue = builder.createBool(loc, mustFree);
  return packageBufferizedExpr(loc, builder, storage, mustFreeValue);
}

/// Helper to extract the storage from a tuple created by packageBufferizedExpr.
/// It assumes no tuples are used as HLFIR operation operands, which is
/// currently enforced by the verifiers that only accept HLFIR value or
/// variable types which do not include tuples.
static mlir::Value getBufferizedExprStorage(mlir::Value bufferizedExpr) {
  auto tupleType = bufferizedExpr.getType().dyn_cast<mlir::TupleType>();
  if (!tupleType)
    return bufferizedExpr;
  assert(tupleType.size() == 2 && "unexpected tuple type");
  if (auto insert = bufferizedExpr.getDefiningOp<fir::InsertValueOp>())
    if (insert.getVal().getType() == tupleType.getType(0))
      return insert.getVal();
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
  if (mold.isArray())
    TODO(loc, "create temps from array mold");
  llvm::SmallVector<mlir::Value> lenParams;
  hlfir::genLengthParameters(loc, builder, mold, lenParams);
  llvm::StringRef tmpName{".tmp"};
  mlir::Value alloca =
      builder.createTemporary(loc, mold.getFortranElementType(), tmpName,
                              /*shape*/ std::nullopt, lenParams);
  auto declareOp = builder.create<hlfir::DeclareOp>(
      loc, alloca, tmpName, /*shapeOrShift*/ mlir::Value{}, lenParams,
      fir::FortranVariableFlagsAttr{});
  mlir::Value falseVal = builder.createBool(loc, false);
  return {hlfir::Entity{declareOp.getBase()}, falseVal};
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
    hlfir::Entity source = hlfir::Entity{adaptor.getVar()};
    auto [temp, cleanup] = createTempFromMold(loc, builder, source);
    builder.create<hlfir::AssignOp>(loc, source, temp);
    mlir::Value bufferizedExpr =
        packageBufferizedExpr(loc, builder, temp, cleanup);
    rewriter.replaceOp(asExpr, bufferizedExpr);
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
    hlfir::Entity lhs{getBufferizedExprStorage(adaptor.getStrings()[0])};
    hlfir::Entity rhs{getBufferizedExprStorage(adaptor.getStrings()[1])};
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
    auto hlfirTempRes = hlfir::genDeclare(loc, builder, res, "tmp",
                                          fir::FortranVariableFlagsAttr{});
    mlir::Value bufferizedExpr =
        packageBufferizedExpr(loc, builder, hlfirTempRes, false);
    rewriter.replaceOp(concat, bufferizedExpr);
    return mlir::success();
  }
};

struct AssociateOpConversion
    : public mlir::OpConversionPattern<hlfir::AssociateOp> {
  using mlir::OpConversionPattern<hlfir::AssociateOp>::OpConversionPattern;
  explicit AssociateOpConversion(mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<hlfir::AssociateOp>{ctx} {}
  mlir::LogicalResult
  matchAndRewrite(hlfir::AssociateOp associate, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = associate->getLoc();
    // If this is the last use of the expression value and this is an hlfir.expr
    // that was bufferized, re-use the storage.
    // Otherwise, create a temp and assign the storage to it.
    mlir::Value bufferizedExpr = getBufferizedExprStorage(adaptor.getSource());
    const bool isTrivialValue = fir::isa_trivial(bufferizedExpr.getType());

    auto replaceWith = [&](mlir::Value hlfirVar, mlir::Value firVar,
                           mlir::Value flag) {
      associate.getResult(0).replaceAllUsesWith(hlfirVar);
      associate.getResult(1).replaceAllUsesWith(firVar);
      associate.getResult(2).replaceAllUsesWith(flag);
      rewriter.replaceOp(associate, {hlfirVar, firVar, flag});
    };

    if (!isTrivialValue && associate.getSource().hasOneUse()) {
      mlir::Value mustFree = getBufferizedExprMustFreeFlag(adaptor.getSource());
      mlir::Value firBase = hlfir::Entity{bufferizedExpr}.getFirBase();
      replaceWith(bufferizedExpr, firBase, mustFree);
      return mlir::success();
    }
    if (isTrivialValue) {
      auto module = associate->getParentOfType<mlir::ModuleOp>();
      fir::FirOpBuilder builder(rewriter, fir::getKindMapping(module));
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

struct EndAssociateOpConversion
    : public mlir::OpConversionPattern<hlfir::EndAssociateOp> {
  using mlir::OpConversionPattern<hlfir::EndAssociateOp>::OpConversionPattern;
  explicit EndAssociateOpConversion(mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<hlfir::EndAssociateOp>{ctx} {}
  mlir::LogicalResult
  matchAndRewrite(hlfir::EndAssociateOp endAssociate, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value mustFree = adaptor.getMustFree();
    if (auto cstMustFree = fir::factory::getIntIfConstant(mustFree))
      if (*cstMustFree == 0) {
        rewriter.eraseOp(endAssociate);
        return mlir::success(); // nothing to do.
      }
    TODO(endAssociate.getLoc(), "conditional free");
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
    patterns.insert<AsExprOpConversion, AssignOpConversion,
                    AssociateOpConversion, ConcatOpConversion,
                    EndAssociateOpConversion, NoReassocOpConversion>(context);
    mlir::ConversionTarget target(*context);
    target.addIllegalOp<hlfir::AssociateOp, hlfir::EndAssociateOp>();
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
