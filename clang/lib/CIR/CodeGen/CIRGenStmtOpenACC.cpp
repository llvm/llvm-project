//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emit OpenACC Stmt nodes as CIR code.
//
//===----------------------------------------------------------------------===//

#include "CIRGenBuilder.h"
#include "CIRGenFunction.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "clang/AST/OpenACCClause.h"
#include "clang/AST/StmtOpenACC.h"

using namespace clang;
using namespace clang::CIRGen;
using namespace cir;
using namespace mlir::acc;

template <typename Op, typename TermOp>
mlir::LogicalResult CIRGenFunction::emitOpenACCOpAssociatedStmt(
    mlir::Location start, mlir::Location end, OpenACCDirectiveKind dirKind,
    SourceLocation dirLoc, llvm::ArrayRef<const OpenACCClause *> clauses,
    const Stmt *associatedStmt) {
  mlir::LogicalResult res = mlir::success();

  llvm::SmallVector<mlir::Type> retTy;
  llvm::SmallVector<mlir::Value> operands;
  auto op = builder.create<Op>(start, retTy, operands);

  emitOpenACCClauses(op, dirKind, dirLoc, clauses);

  {
    mlir::Block &block = op.getRegion().emplaceBlock();
    mlir::OpBuilder::InsertionGuard guardCase(builder);
    builder.setInsertionPointToEnd(&block);

    LexicalScope ls{*this, start, builder.getInsertionBlock()};
    res = emitStmt(associatedStmt, /*useCurrentScope=*/true);

    builder.create<TermOp>(end);
  }
  return res;
}

namespace {
template <typename Op> struct CombinedType;
template <> struct CombinedType<ParallelOp> {
  static constexpr mlir::acc::CombinedConstructsType value =
      mlir::acc::CombinedConstructsType::ParallelLoop;
};
template <> struct CombinedType<SerialOp> {
  static constexpr mlir::acc::CombinedConstructsType value =
      mlir::acc::CombinedConstructsType::SerialLoop;
};
template <> struct CombinedType<KernelsOp> {
  static constexpr mlir::acc::CombinedConstructsType value =
      mlir::acc::CombinedConstructsType::KernelsLoop;
};
} // namespace

template <typename Op, typename TermOp>
mlir::LogicalResult CIRGenFunction::emitOpenACCOpCombinedConstruct(
    mlir::Location start, mlir::Location end, OpenACCDirectiveKind dirKind,
    SourceLocation dirLoc, llvm::ArrayRef<const OpenACCClause *> clauses,
    const Stmt *loopStmt) {
  mlir::LogicalResult res = mlir::success();

  llvm::SmallVector<mlir::Type> retTy;
  llvm::SmallVector<mlir::Value> operands;

  auto computeOp = builder.create<Op>(start, retTy, operands);
  computeOp.setCombinedAttr(builder.getUnitAttr());
  mlir::acc::LoopOp loopOp;

  // First, emit the bodies of both operations, with the loop inside the body of
  // the combined construct.
  {
    mlir::Block &block = computeOp.getRegion().emplaceBlock();
    mlir::OpBuilder::InsertionGuard guardCase(builder);
    builder.setInsertionPointToEnd(&block);

    LexicalScope ls{*this, start, builder.getInsertionBlock()};
    auto loopOp = builder.create<LoopOp>(start, retTy, operands);
    loopOp.setCombinedAttr(mlir::acc::CombinedConstructsTypeAttr::get(
        builder.getContext(), CombinedType<Op>::value));

    {
      mlir::Block &innerBlock = loopOp.getRegion().emplaceBlock();
      mlir::OpBuilder::InsertionGuard guardCase(builder);
      builder.setInsertionPointToEnd(&innerBlock);

      LexicalScope ls{*this, start, builder.getInsertionBlock()};
      res = emitStmt(loopStmt, /*useCurrentScope=*/true);

      builder.create<mlir::acc::YieldOp>(end);
    }

    emitOpenACCClauses(computeOp, loopOp, dirKind, dirLoc, clauses);

    updateLoopOpParallelism(loopOp, /*isOrphan=*/false, dirKind);

    builder.create<TermOp>(end);
  }

  return res;
}

template <typename Op>
Op CIRGenFunction::emitOpenACCOp(
    mlir::Location start, OpenACCDirectiveKind dirKind, SourceLocation dirLoc,
    llvm::ArrayRef<const OpenACCClause *> clauses) {
  llvm::SmallVector<mlir::Type> retTy;
  llvm::SmallVector<mlir::Value> operands;
  auto op = builder.create<Op>(start, retTy, operands);

  emitOpenACCClauses(op, dirKind, dirLoc, clauses);
  return op;
}

mlir::LogicalResult
CIRGenFunction::emitOpenACCComputeConstruct(const OpenACCComputeConstruct &s) {
  mlir::Location start = getLoc(s.getSourceRange().getBegin());
  mlir::Location end = getLoc(s.getSourceRange().getEnd());

  switch (s.getDirectiveKind()) {
  case OpenACCDirectiveKind::Parallel:
    return emitOpenACCOpAssociatedStmt<ParallelOp, mlir::acc::YieldOp>(
        start, end, s.getDirectiveKind(), s.getDirectiveLoc(), s.clauses(),
        s.getStructuredBlock());
  case OpenACCDirectiveKind::Serial:
    return emitOpenACCOpAssociatedStmt<SerialOp, mlir::acc::YieldOp>(
        start, end, s.getDirectiveKind(), s.getDirectiveLoc(), s.clauses(),
        s.getStructuredBlock());
  case OpenACCDirectiveKind::Kernels:
    return emitOpenACCOpAssociatedStmt<KernelsOp, mlir::acc::TerminatorOp>(
        start, end, s.getDirectiveKind(), s.getDirectiveLoc(), s.clauses(),
        s.getStructuredBlock());
  default:
    llvm_unreachable("invalid compute construct kind");
  }
}

mlir::LogicalResult
CIRGenFunction::emitOpenACCDataConstruct(const OpenACCDataConstruct &s) {
  mlir::Location start = getLoc(s.getSourceRange().getBegin());
  mlir::Location end = getLoc(s.getSourceRange().getEnd());

  return emitOpenACCOpAssociatedStmt<DataOp, mlir::acc::TerminatorOp>(
      start, end, s.getDirectiveKind(), s.getDirectiveLoc(), s.clauses(),
      s.getStructuredBlock());
}

mlir::LogicalResult
CIRGenFunction::emitOpenACCInitConstruct(const OpenACCInitConstruct &s) {
  mlir::Location start = getLoc(s.getSourceRange().getBegin());
  emitOpenACCOp<InitOp>(start, s.getDirectiveKind(), s.getDirectiveLoc(),
                               s.clauses());
  return mlir::success();
}

mlir::LogicalResult
CIRGenFunction::emitOpenACCSetConstruct(const OpenACCSetConstruct &s) {
  mlir::Location start = getLoc(s.getSourceRange().getBegin());
  emitOpenACCOp<SetOp>(start, s.getDirectiveKind(), s.getDirectiveLoc(),
                              s.clauses());
  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::emitOpenACCShutdownConstruct(
    const OpenACCShutdownConstruct &s) {
  mlir::Location start = getLoc(s.getSourceRange().getBegin());
  emitOpenACCOp<ShutdownOp>(start, s.getDirectiveKind(),
                                   s.getDirectiveLoc(), s.clauses());
  return mlir::success();
}

mlir::LogicalResult
CIRGenFunction::emitOpenACCWaitConstruct(const OpenACCWaitConstruct &s) {
  mlir::Location start = getLoc(s.getSourceRange().getBegin());
  auto waitOp = emitOpenACCOp<WaitOp>(start, s.getDirectiveKind(),
                                   s.getDirectiveLoc(), s.clauses());

  auto createIntExpr = [this](const Expr *intExpr) {
    mlir::Value expr = emitScalarExpr(intExpr);
    mlir::Location exprLoc = cgm.getLoc(intExpr->getBeginLoc());

    mlir::IntegerType targetType = mlir::IntegerType::get(
        &getMLIRContext(), getContext().getIntWidth(intExpr->getType()),
        intExpr->getType()->isSignedIntegerOrEnumerationType()
            ? mlir::IntegerType::SignednessSemantics::Signed
            : mlir::IntegerType::SignednessSemantics::Unsigned);

    auto conversionOp = builder.create<mlir::UnrealizedConversionCastOp>(
        exprLoc, targetType, expr);
    return conversionOp.getResult(0);
  };

  // Emit the correct 'wait' clauses.
  {
    mlir::OpBuilder::InsertionGuard guardCase(builder);
    builder.setInsertionPoint(waitOp);

    if (s.hasDevNumExpr())
      waitOp.getWaitDevnumMutable().append(createIntExpr(s.getDevNumExpr()));

    for (Expr *QueueExpr : s.getQueueIdExprs())
      waitOp.getWaitOperandsMutable().append(createIntExpr(QueueExpr));
  }

  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::emitOpenACCCombinedConstruct(
    const OpenACCCombinedConstruct &s) {
  mlir::Location start = getLoc(s.getSourceRange().getBegin());
  mlir::Location end = getLoc(s.getSourceRange().getEnd());

  switch (s.getDirectiveKind()) {
  case OpenACCDirectiveKind::ParallelLoop:
    return emitOpenACCOpCombinedConstruct<ParallelOp, mlir::acc::YieldOp>(
        start, end, s.getDirectiveKind(), s.getDirectiveLoc(), s.clauses(),
        s.getLoop());
  case OpenACCDirectiveKind::SerialLoop:
    return emitOpenACCOpCombinedConstruct<SerialOp, mlir::acc::YieldOp>(
        start, end, s.getDirectiveKind(), s.getDirectiveLoc(), s.clauses(),
        s.getLoop());
  case OpenACCDirectiveKind::KernelsLoop:
    return emitOpenACCOpCombinedConstruct<KernelsOp, mlir::acc::TerminatorOp>(
        start, end, s.getDirectiveKind(), s.getDirectiveLoc(), s.clauses(),
        s.getLoop());
  default:
    llvm_unreachable("invalid compute construct kind");
  }
}

mlir::LogicalResult CIRGenFunction::emitOpenACCHostDataConstruct(
    const OpenACCHostDataConstruct &s) {
  mlir::Location start = getLoc(s.getSourceRange().getBegin());
  mlir::Location end = getLoc(s.getSourceRange().getEnd());

  return emitOpenACCOpAssociatedStmt<HostDataOp, mlir::acc::TerminatorOp>(
      start, end, s.getDirectiveKind(), s.getDirectiveLoc(), s.clauses(),
      s.getStructuredBlock());
}

mlir::LogicalResult CIRGenFunction::emitOpenACCEnterDataConstruct(
    const OpenACCEnterDataConstruct &s) {
  cgm.errorNYI(s.getSourceRange(), "OpenACC EnterData Construct");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOpenACCExitDataConstruct(
    const OpenACCExitDataConstruct &s) {
  cgm.errorNYI(s.getSourceRange(), "OpenACC ExitData Construct");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOpenACCUpdateConstruct(const OpenACCUpdateConstruct &s) {
  cgm.errorNYI(s.getSourceRange(), "OpenACC Update Construct");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOpenACCAtomicConstruct(const OpenACCAtomicConstruct &s) {
  cgm.errorNYI(s.getSourceRange(), "OpenACC Atomic Construct");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOpenACCCacheConstruct(const OpenACCCacheConstruct &s) {
  cgm.errorNYI(s.getSourceRange(), "OpenACC Cache Construct");
  return mlir::failure();
}
