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
  auto op = Op::create(builder, start, retTy, operands);

  emitOpenACCClauses(op, dirKind, dirLoc, clauses);

  {
    mlir::Block &block = op.getRegion().emplaceBlock();
    mlir::OpBuilder::InsertionGuard guardCase(builder);
    builder.setInsertionPointToEnd(&block);

    LexicalScope ls{*this, start, builder.getInsertionBlock()};
    res = emitStmt(associatedStmt, /*useCurrentScope=*/true);

    TermOp::create(builder, end);
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

  auto computeOp = Op::create(builder, start, retTy, operands);
  computeOp.setCombinedAttr(builder.getUnitAttr());
  mlir::acc::LoopOp loopOp;

  // First, emit the bodies of both operations, with the loop inside the body of
  // the combined construct.
  {
    mlir::Block &block = computeOp.getRegion().emplaceBlock();
    mlir::OpBuilder::InsertionGuard guardCase(builder);
    builder.setInsertionPointToEnd(&block);

    LexicalScope ls{*this, start, builder.getInsertionBlock()};
    auto loopOp = LoopOp::create(builder, start, retTy, operands);
    loopOp.setCombinedAttr(mlir::acc::CombinedConstructsTypeAttr::get(
        builder.getContext(), CombinedType<Op>::value));

    {
      mlir::Block &innerBlock = loopOp.getRegion().emplaceBlock();
      mlir::OpBuilder::InsertionGuard guardCase(builder);
      builder.setInsertionPointToEnd(&innerBlock);

      LexicalScope ls{*this, start, builder.getInsertionBlock()};
      ActiveOpenACCLoopRAII activeLoop{*this, &loopOp};

      res = emitStmt(loopStmt, /*useCurrentScope=*/true);

      mlir::acc::YieldOp::create(builder, end);
    }

    emitOpenACCClauses(computeOp, loopOp, dirKind, dirLoc, clauses);

    updateLoopOpParallelism(loopOp, /*isOrphan=*/false, dirKind);

    TermOp::create(builder, end);
  }

  return res;
}

template <typename Op>
Op CIRGenFunction::emitOpenACCOp(
    mlir::Location start, OpenACCDirectiveKind dirKind, SourceLocation dirLoc,
    llvm::ArrayRef<const OpenACCClause *> clauses) {
  llvm::SmallVector<mlir::Type> retTy;
  llvm::SmallVector<mlir::Value> operands;
  auto op = Op::create(builder, start, retTy, operands);

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

    auto conversionOp = mlir::UnrealizedConversionCastOp::create(
        builder, exprLoc, targetType, expr);
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
  mlir::Location start = getLoc(s.getSourceRange().getBegin());
  emitOpenACCOp<EnterDataOp>(start, s.getDirectiveKind(), s.getDirectiveLoc(),
                             s.clauses());
  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::emitOpenACCExitDataConstruct(
    const OpenACCExitDataConstruct &s) {
  mlir::Location start = getLoc(s.getSourceRange().getBegin());
  emitOpenACCOp<ExitDataOp>(start, s.getDirectiveKind(), s.getDirectiveLoc(),
                            s.clauses());
  return mlir::success();
}

mlir::LogicalResult
CIRGenFunction::emitOpenACCUpdateConstruct(const OpenACCUpdateConstruct &s) {
  mlir::Location start = getLoc(s.getSourceRange().getBegin());
  emitOpenACCOp<UpdateOp>(start, s.getDirectiveKind(), s.getDirectiveLoc(),
                          s.clauses());
  return mlir::success();
}

mlir::LogicalResult
CIRGenFunction::emitOpenACCCacheConstruct(const OpenACCCacheConstruct &s) {
  // The 'cache' directive 'may' be at the top of a loop by standard, but
  // doesn't have to be. Additionally, there is nothing that requires this be a
  // loop affected by an OpenACC pragma. Sema doesn't do any level of
  // enforcement here, since it isn't particularly valuable to do so thanks to
  // that. Instead, we treat cache as a 'noop' if there is no acc.loop to apply
  // it to.
  if (!activeLoopOp)
    return mlir::success();

  mlir::acc::LoopOp loopOp = *activeLoopOp;

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(loopOp);

  for (const Expr *var : s.getVarList()) {
    CIRGenFunction::OpenACCDataOperandInfo opInfo =
        getOpenACCDataOperandInfo(var);

    auto cacheOp = CacheOp::create(builder, opInfo.beginLoc, opInfo.varValue,
                                   /*structured=*/false, /*implicit=*/false,
                                   opInfo.name, opInfo.bounds);

    loopOp.getCacheOperandsMutable().append(cacheOp.getResult());
  }

  return mlir::success();
}

mlir::LogicalResult
CIRGenFunction::emitOpenACCAtomicConstruct(const OpenACCAtomicConstruct &s) {
  // For now, we are only support 'read'/'write', so diagnose. We can switch on
  // the kind later once we start implementing the other 2 forms. While we
  if (s.getAtomicKind() != OpenACCAtomicKind::Read &&
      s.getAtomicKind() != OpenACCAtomicKind::Write) {
    cgm.errorNYI(s.getSourceRange(), "OpenACC Atomic Construct");
    return mlir::failure();
  }

  // While Atomic is an 'associated statement' construct, it 'steals' the
  // expression it is associated with rather than emitting it inside of it.  So
  // it has custom emit logic.
  mlir::Location start = getLoc(s.getSourceRange().getBegin());
  OpenACCAtomicConstruct::StmtInfo inf = s.getAssociatedStmtInfo();

  switch (s.getAtomicKind()) {
  case OpenACCAtomicKind::None:
  case OpenACCAtomicKind::Update:
  case OpenACCAtomicKind::Capture:
    llvm_unreachable("Unimplemented atomic construct type, should have "
                     "diagnosed/returned above");
    return mlir::failure();
  case OpenACCAtomicKind::Read: {

    // Atomic 'read' only permits 'v = x', where v and x are both scalar L
    // values. The getAssociatedStmtInfo strips off implicit casts, which
    // includes implicit conversions and L-to-R-Value conversions, so we can
    // just emit it as an L value.  The Flang implementation has no problem with
    // different types, so it appears that the dialect can handle the
    // conversions.
    mlir::Value v = emitLValue(inf.V).getPointer();
    mlir::Value x = emitLValue(inf.X).getPointer();
    mlir::Type resTy = convertType(inf.V->getType());
    auto op = mlir::acc::AtomicReadOp::create(builder, start, x, v, resTy,
                                              /*ifCond=*/{});
    emitOpenACCClauses(op, s.getDirectiveKind(), s.getDirectiveLoc(),
                       s.clauses());
    return mlir::success();
  }
  case OpenACCAtomicKind::Write: {
    mlir::Value x = emitLValue(inf.X).getPointer();
    mlir::Value expr = emitAnyExpr(inf.RefExpr).getValue();
    auto op = mlir::acc::AtomicWriteOp::create(builder, start, x, expr,
                                               /*ifCond=*/{});
    emitOpenACCClauses(op, s.getDirectiveKind(), s.getDirectiveLoc(),
                       s.clauses());
    return mlir::success();
  }
  }

  llvm_unreachable("unknown OpenACC atomic kind");
}
