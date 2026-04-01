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
#include "aiir/Dialect/OpenACC/OpenACC.h"
#include "clang/AST/OpenACCClause.h"
#include "clang/AST/StmtOpenACC.h"

using namespace clang;
using namespace clang::CIRGen;
using namespace cir;
using namespace aiir::acc;

template <typename Op, typename TermOp>
aiir::LogicalResult CIRGenFunction::emitOpenACCOpAssociatedStmt(
    aiir::Location start, aiir::Location end, OpenACCDirectiveKind dirKind,
    llvm::ArrayRef<const OpenACCClause *> clauses, const Stmt *associatedStmt) {
  aiir::LogicalResult res = aiir::success();

  llvm::SmallVector<aiir::Type> retTy;
  llvm::SmallVector<aiir::Value> operands;
  auto op = Op::create(builder, start, retTy, operands);

  emitOpenACCClauses(op, dirKind, clauses);

  {
    aiir::Block &block = op.getRegion().emplaceBlock();
    aiir::OpBuilder::InsertionGuard guardCase(builder);
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
  static constexpr aiir::acc::CombinedConstructsType value =
      aiir::acc::CombinedConstructsType::ParallelLoop;
};
template <> struct CombinedType<SerialOp> {
  static constexpr aiir::acc::CombinedConstructsType value =
      aiir::acc::CombinedConstructsType::SerialLoop;
};
template <> struct CombinedType<KernelsOp> {
  static constexpr aiir::acc::CombinedConstructsType value =
      aiir::acc::CombinedConstructsType::KernelsLoop;
};
} // namespace

template <typename Op, typename TermOp>
aiir::LogicalResult CIRGenFunction::emitOpenACCOpCombinedConstruct(
    aiir::Location start, aiir::Location end, OpenACCDirectiveKind dirKind,
    llvm::ArrayRef<const OpenACCClause *> clauses, const Stmt *loopStmt) {
  aiir::LogicalResult res = aiir::success();

  llvm::SmallVector<aiir::Type> retTy;
  llvm::SmallVector<aiir::Value> operands;

  auto computeOp = Op::create(builder, start, retTy, operands);
  computeOp.setCombinedAttr(builder.getUnitAttr());
  aiir::acc::LoopOp loopOp;

  // First, emit the bodies of both operations, with the loop inside the body of
  // the combined construct.
  {
    aiir::Block &block = computeOp.getRegion().emplaceBlock();
    aiir::OpBuilder::InsertionGuard guardCase(builder);
    builder.setInsertionPointToEnd(&block);

    LexicalScope ls{*this, start, builder.getInsertionBlock()};
    auto loopOp = LoopOp::create(builder, start, retTy, operands);
    loopOp.setCombinedAttr(aiir::acc::CombinedConstructsTypeAttr::get(
        builder.getContext(), CombinedType<Op>::value));

    {
      aiir::Block &innerBlock = loopOp.getRegion().emplaceBlock();
      aiir::OpBuilder::InsertionGuard guardCase(builder);
      builder.setInsertionPointToEnd(&innerBlock);

      LexicalScope ls{*this, start, builder.getInsertionBlock()};
      ActiveOpenACCLoopRAII activeLoop{*this, &loopOp};

      res = emitStmt(loopStmt, /*useCurrentScope=*/true);

      aiir::acc::YieldOp::create(builder, end);
    }

    emitOpenACCClauses(computeOp, loopOp, dirKind, clauses);

    updateLoopOpParallelism(loopOp, /*isOrphan=*/false, dirKind);

    TermOp::create(builder, end);
  }

  return res;
}

template <typename Op>
Op CIRGenFunction::emitOpenACCOp(
    aiir::Location start, OpenACCDirectiveKind dirKind,
    llvm::ArrayRef<const OpenACCClause *> clauses) {
  llvm::SmallVector<aiir::Type> retTy;
  llvm::SmallVector<aiir::Value> operands;
  auto op = Op::create(builder, start, retTy, operands);

  emitOpenACCClauses(op, dirKind, clauses);
  return op;
}

aiir::LogicalResult
CIRGenFunction::emitOpenACCComputeConstruct(const OpenACCComputeConstruct &s) {
  aiir::Location start = getLoc(s.getSourceRange().getBegin());
  aiir::Location end = getLoc(s.getSourceRange().getEnd());

  switch (s.getDirectiveKind()) {
  case OpenACCDirectiveKind::Parallel:
    return emitOpenACCOpAssociatedStmt<ParallelOp, aiir::acc::YieldOp>(
        start, end, s.getDirectiveKind(), s.clauses(), s.getStructuredBlock());
  case OpenACCDirectiveKind::Serial:
    return emitOpenACCOpAssociatedStmt<SerialOp, aiir::acc::YieldOp>(
        start, end, s.getDirectiveKind(), s.clauses(), s.getStructuredBlock());
  case OpenACCDirectiveKind::Kernels:
    return emitOpenACCOpAssociatedStmt<KernelsOp, aiir::acc::TerminatorOp>(
        start, end, s.getDirectiveKind(), s.clauses(), s.getStructuredBlock());
  default:
    llvm_unreachable("invalid compute construct kind");
  }
}

aiir::LogicalResult
CIRGenFunction::emitOpenACCDataConstruct(const OpenACCDataConstruct &s) {
  aiir::Location start = getLoc(s.getSourceRange().getBegin());
  aiir::Location end = getLoc(s.getSourceRange().getEnd());

  return emitOpenACCOpAssociatedStmt<DataOp, aiir::acc::TerminatorOp>(
      start, end, s.getDirectiveKind(), s.clauses(), s.getStructuredBlock());
}

aiir::LogicalResult
CIRGenFunction::emitOpenACCInitConstruct(const OpenACCInitConstruct &s) {
  aiir::Location start = getLoc(s.getSourceRange().getBegin());
  emitOpenACCOp<InitOp>(start, s.getDirectiveKind(), s.clauses());
  return aiir::success();
}

aiir::LogicalResult
CIRGenFunction::emitOpenACCSetConstruct(const OpenACCSetConstruct &s) {
  aiir::Location start = getLoc(s.getSourceRange().getBegin());
  emitOpenACCOp<SetOp>(start, s.getDirectiveKind(), s.clauses());
  return aiir::success();
}

aiir::LogicalResult CIRGenFunction::emitOpenACCShutdownConstruct(
    const OpenACCShutdownConstruct &s) {
  aiir::Location start = getLoc(s.getSourceRange().getBegin());
  emitOpenACCOp<ShutdownOp>(start, s.getDirectiveKind(), s.clauses());
  return aiir::success();
}

aiir::LogicalResult
CIRGenFunction::emitOpenACCWaitConstruct(const OpenACCWaitConstruct &s) {
  aiir::Location start = getLoc(s.getSourceRange().getBegin());
  auto waitOp = emitOpenACCOp<WaitOp>(start, s.getDirectiveKind(), s.clauses());

  auto createIntExpr = [this](const Expr *intExpr) {
    aiir::Value expr = emitScalarExpr(intExpr);
    aiir::Location exprLoc = cgm.getLoc(intExpr->getBeginLoc());

    aiir::IntegerType targetType = aiir::IntegerType::get(
        &getAIIRContext(), getContext().getIntWidth(intExpr->getType()),
        intExpr->getType()->isSignedIntegerOrEnumerationType()
            ? aiir::IntegerType::SignednessSemantics::Signed
            : aiir::IntegerType::SignednessSemantics::Unsigned);

    auto conversionOp = aiir::UnrealizedConversionCastOp::create(
        builder, exprLoc, targetType, expr);
    return conversionOp.getResult(0);
  };

  // Emit the correct 'wait' clauses.
  {
    aiir::OpBuilder::InsertionGuard guardCase(builder);
    builder.setInsertionPoint(waitOp);

    if (s.hasDevNumExpr())
      waitOp.getWaitDevnumMutable().append(createIntExpr(s.getDevNumExpr()));

    for (Expr *QueueExpr : s.getQueueIdExprs())
      waitOp.getWaitOperandsMutable().append(createIntExpr(QueueExpr));
  }

  return aiir::success();
}

aiir::LogicalResult CIRGenFunction::emitOpenACCCombinedConstruct(
    const OpenACCCombinedConstruct &s) {
  aiir::Location start = getLoc(s.getSourceRange().getBegin());
  aiir::Location end = getLoc(s.getSourceRange().getEnd());

  switch (s.getDirectiveKind()) {
  case OpenACCDirectiveKind::ParallelLoop:
    return emitOpenACCOpCombinedConstruct<ParallelOp, aiir::acc::YieldOp>(
        start, end, s.getDirectiveKind(), s.clauses(), s.getLoop());
  case OpenACCDirectiveKind::SerialLoop:
    return emitOpenACCOpCombinedConstruct<SerialOp, aiir::acc::YieldOp>(
        start, end, s.getDirectiveKind(), s.clauses(), s.getLoop());
  case OpenACCDirectiveKind::KernelsLoop:
    return emitOpenACCOpCombinedConstruct<KernelsOp, aiir::acc::TerminatorOp>(
        start, end, s.getDirectiveKind(), s.clauses(), s.getLoop());
  default:
    llvm_unreachable("invalid compute construct kind");
  }
}

aiir::LogicalResult CIRGenFunction::emitOpenACCHostDataConstruct(
    const OpenACCHostDataConstruct &s) {
  aiir::Location start = getLoc(s.getSourceRange().getBegin());
  aiir::Location end = getLoc(s.getSourceRange().getEnd());

  return emitOpenACCOpAssociatedStmt<HostDataOp, aiir::acc::TerminatorOp>(
      start, end, s.getDirectiveKind(), s.clauses(), s.getStructuredBlock());
}

aiir::LogicalResult CIRGenFunction::emitOpenACCEnterDataConstruct(
    const OpenACCEnterDataConstruct &s) {
  aiir::Location start = getLoc(s.getSourceRange().getBegin());
  emitOpenACCOp<EnterDataOp>(start, s.getDirectiveKind(), s.clauses());
  return aiir::success();
}

aiir::LogicalResult CIRGenFunction::emitOpenACCExitDataConstruct(
    const OpenACCExitDataConstruct &s) {
  aiir::Location start = getLoc(s.getSourceRange().getBegin());
  emitOpenACCOp<ExitDataOp>(start, s.getDirectiveKind(), s.clauses());
  return aiir::success();
}

aiir::LogicalResult
CIRGenFunction::emitOpenACCUpdateConstruct(const OpenACCUpdateConstruct &s) {
  aiir::Location start = getLoc(s.getSourceRange().getBegin());
  emitOpenACCOp<UpdateOp>(start, s.getDirectiveKind(), s.clauses());
  return aiir::success();
}

aiir::LogicalResult
CIRGenFunction::emitOpenACCCacheConstruct(const OpenACCCacheConstruct &s) {
  // The 'cache' directive 'may' be at the top of a loop by standard, but
  // doesn't have to be. Additionally, there is nothing that requires this be a
  // loop affected by an OpenACC pragma. Sema doesn't do any level of
  // enforcement here, since it isn't particularly valuable to do so thanks to
  // that. Instead, we treat cache as a 'noop' if there is no acc.loop to apply
  // it to.
  if (!activeLoopOp)
    return aiir::success();

  aiir::acc::LoopOp loopOp = *activeLoopOp;

  aiir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(loopOp);

  for (const Expr *var : s.getVarList()) {
    CIRGenFunction::OpenACCDataOperandInfo opInfo =
        getOpenACCDataOperandInfo(var);

    auto cacheOp = CacheOp::create(builder, opInfo.beginLoc, opInfo.varValue,
                                   /*structured=*/false, /*implicit=*/false,
                                   opInfo.name, opInfo.bounds);

    loopOp.getCacheOperandsMutable().append(cacheOp.getResult());
  }

  return aiir::success();
}

const VarDecl *getLValueDecl(const Expr *e) {
  // We are going to assume that after stripping implicit casts, that the LValue
  // is just a DRE around the var-decl.

  e = e->IgnoreImpCasts();

  const auto *dre = cast<DeclRefExpr>(e);
  return cast<VarDecl>(dre->getDecl());
}

static aiir::acc::AtomicReadOp
emitAtomicRead(CIRGenFunction &cgf, CIRGenBuilderTy &builder,
               aiir::Location start,
               const OpenACCAtomicConstruct::SingleStmtInfo &inf) {
  // Atomic 'read' only permits 'v = x', where v and x are both scalar L
  // values. The getAssociatedStmtInfo strips off implicit casts, which
  // includes implicit conversions and L-to-R-Value conversions, so we can
  // just emit it as an L value.  The Flang implementation has no problem with
  // different types, so it appears that the dialect can handle the
  // conversions.
  aiir::Value v = cgf.emitLValue(inf.V).getPointer();
  aiir::Value x = cgf.emitLValue(inf.X).getPointer();
  aiir::Type resTy = cgf.convertType(inf.V->getType());
  return aiir::acc::AtomicReadOp::create(builder, start, x, v, resTy,
                                         /*ifCond=*/{});
}

static aiir::acc::AtomicWriteOp
emitAtomicWrite(CIRGenFunction &cgf, CIRGenBuilderTy &builder,
                aiir::Location start,
                const OpenACCAtomicConstruct::SingleStmtInfo &inf) {
  aiir::Value x = cgf.emitLValue(inf.X).getPointer();
  aiir::Value expr = cgf.emitAnyExpr(inf.RefExpr).getValue();
  return aiir::acc::AtomicWriteOp::create(builder, start, x, expr,
                                          /*ifCond=*/{});
}

static std::pair<aiir::LogicalResult, aiir::acc::AtomicUpdateOp>
emitAtomicUpdate(CIRGenFunction &cgf, CIRGenBuilderTy &builder,
                 aiir::Location start, aiir::Location end,
                 const OpenACCAtomicConstruct::SingleStmtInfo &inf) {
  aiir::Value x = cgf.emitLValue(inf.X).getPointer();
  auto op = aiir::acc::AtomicUpdateOp::create(builder, start, x, /*ifCond=*/{});

  aiir::LogicalResult res = aiir::success();
  {
    aiir::OpBuilder::InsertionGuard guardCase(builder);
    aiir::Type argTy = cast<cir::PointerType>(x.getType()).getPointee();
    std::array<aiir::Type, 1> recipeType{argTy};
    std::array<aiir::Location, 1> recipeLoc{start};
    auto *recipeBlock = builder.createBlock(
        &op.getRegion(), op.getRegion().end(), recipeType, recipeLoc);
    builder.setInsertionPointToEnd(recipeBlock);
    // Since we have an initial value that we know is a scalar type, we can
    // just emit the entire statement here after sneaking-in our 'alloca' in
    // the right place, then loading out of it. Flang does a lot less work
    // (probably does its own emitting!), but we have more complicated AST
    // nodes to worry about, so we can just count on opt to remove the extra
    // alloca/load/store set.
    auto alloca = cir::AllocaOp::create(
        builder, start, x.getType(), argTy, "x_var",
        cgf.cgm.getSize(
            cgf.getContext().getTypeAlignInChars(inf.X->getType())));

    alloca.setInitAttr(builder.getUnitAttr());
    builder.CIRBaseBuilderTy::createStore(start, recipeBlock->getArgument(0),
                                          alloca);

    const VarDecl *xval = getLValueDecl(inf.X);
    CIRGenFunction::DeclMapRevertingRAII declMapRAII{cgf, xval};
    cgf.replaceAddrOfLocalVar(
        xval, Address{alloca, argTy, cgf.getContext().getDeclAlign(xval)});

    res = cgf.emitStmt(inf.WholeExpr, /*useCurrentScope=*/true);

    auto load = cir::LoadOp::create(builder, start, {alloca});
    aiir::acc::YieldOp::create(builder, end, {load});
  }

  return {res, op};
}

aiir::LogicalResult
CIRGenFunction::emitOpenACCAtomicConstruct(const OpenACCAtomicConstruct &s) {
  // While Atomic is an 'associated statement' construct, it 'steals' the
  // expression it is associated with rather than emitting it inside of it.  So
  // it has custom emit logic.
  aiir::Location start = getLoc(s.getSourceRange().getBegin());
  aiir::Location end = getLoc(s.getSourceRange().getEnd());
  OpenACCAtomicConstruct::StmtInfo inf = s.getAssociatedStmtInfo();

  switch (s.getAtomicKind()) {
  case OpenACCAtomicKind::Read: {
    assert(inf.Form == OpenACCAtomicConstruct::StmtInfo::StmtForm::Read);
    aiir::acc::AtomicReadOp op =
        emitAtomicRead(*this, builder, start, inf.First);
    emitOpenACCClauses(op, s.getDirectiveKind(), s.clauses());
    return aiir::success();
  }
  case OpenACCAtomicKind::Write: {
    assert(inf.Form == OpenACCAtomicConstruct::StmtInfo::StmtForm::Write);
    auto op = emitAtomicWrite(*this, builder, start, inf.First);
    emitOpenACCClauses(op, s.getDirectiveKind(), s.clauses());
    return aiir::success();
  }
  case OpenACCAtomicKind::None:
  case OpenACCAtomicKind::Update: {
    assert(inf.Form == OpenACCAtomicConstruct::StmtInfo::StmtForm::Update);
    auto [res, op] = emitAtomicUpdate(*this, builder, start, end, inf.First);
    emitOpenACCClauses(op, s.getDirectiveKind(), s.clauses());
    return res;
  }
  case OpenACCAtomicKind::Capture: {
    // Atomic-capture is made up of two statements, either an update = read,
    // read + update, or read + write.  As a result, the IR represents the
    // capture region as having those two 'inside' of it.
    auto op = aiir::acc::AtomicCaptureOp::create(builder, start, /*ifCond=*/{});
    emitOpenACCClauses(op, s.getDirectiveKind(), s.clauses());
    aiir::LogicalResult res = aiir::success();
    {
      aiir::OpBuilder::InsertionGuard guardCase(builder);

      aiir::Block *block =
          builder.createBlock(&op.getRegion(), op.getRegion().end(), {}, {});

      builder.setInsertionPointToStart(block);

      auto terminator = aiir::acc::TerminatorOp::create(builder, end);

      // The AtomicCaptureOp only permits the two acc.atomic.* operations inside
      // of it, so all other parts of the expression need to be emitted before
      // the AtomicCaptureOp, then moved into place.
      builder.setInsertionPoint(op);

      switch (inf.Form) {
      default:
        llvm_unreachable("invalid form for Capture");
      case OpenACCAtomicConstruct::StmtInfo::StmtForm::ReadWrite: {
        aiir::acc::AtomicReadOp first =
            emitAtomicRead(*this, builder, start, inf.First);
        aiir::acc::AtomicWriteOp second =
            emitAtomicWrite(*this, builder, start, inf.Second);

        first->moveBefore(terminator);
        second->moveBefore(terminator);
        break;
      }
      case OpenACCAtomicConstruct::StmtInfo::StmtForm::ReadUpdate: {
        aiir::acc::AtomicReadOp first =
            emitAtomicRead(*this, builder, start, inf.First);
        auto [this_res, second] =
            emitAtomicUpdate(*this, builder, start, end, inf.Second);
        res = this_res;

        first->moveBefore(terminator);
        second->moveBefore(terminator);
        break;
      }
      case OpenACCAtomicConstruct::StmtInfo::StmtForm::UpdateRead: {
        auto [this_res, first] =
            emitAtomicUpdate(*this, builder, start, end, inf.First);
        res = this_res;
        aiir::acc::AtomicReadOp second =
            emitAtomicRead(*this, builder, start, inf.Second);

        first->moveBefore(terminator);
        second->moveBefore(terminator);
        break;
      }
      }
    }
    return res;
  }
  }

  llvm_unreachable("unknown OpenACC atomic kind");
}
