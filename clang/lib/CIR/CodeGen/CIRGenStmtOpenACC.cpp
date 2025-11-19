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

const VarDecl *getLValueDecl(const Expr *e) {
  // We are going to assume that after stripping implicit casts, that the LValue
  // is just a DRE around the var-decl.

  e = e->IgnoreImpCasts();

  const auto *dre = cast<DeclRefExpr>(e);
  return cast<VarDecl>(dre->getDecl());
}

static mlir::acc::AtomicReadOp
emitAtomicRead(CIRGenFunction &cgf, CIRGenBuilderTy &builder,
               mlir::Location start,
               const OpenACCAtomicConstruct::SingleStmtInfo &inf) {
  // Atomic 'read' only permits 'v = x', where v and x are both scalar L
  // values. The getAssociatedStmtInfo strips off implicit casts, which
  // includes implicit conversions and L-to-R-Value conversions, so we can
  // just emit it as an L value.  The Flang implementation has no problem with
  // different types, so it appears that the dialect can handle the
  // conversions.
  mlir::Value v = cgf.emitLValue(inf.V).getPointer();
  mlir::Value x = cgf.emitLValue(inf.X).getPointer();
  mlir::Type resTy = cgf.convertType(inf.V->getType());
  return mlir::acc::AtomicReadOp::create(builder, start, x, v, resTy,
                                         /*ifCond=*/{});
}

static mlir::acc::AtomicWriteOp
emitAtomicWrite(CIRGenFunction &cgf, CIRGenBuilderTy &builder,
                mlir::Location start,
                const OpenACCAtomicConstruct::SingleStmtInfo &inf) {
  mlir::Value x = cgf.emitLValue(inf.X).getPointer();
  mlir::Value expr = cgf.emitAnyExpr(inf.RefExpr).getValue();
  return mlir::acc::AtomicWriteOp::create(builder, start, x, expr,
                                          /*ifCond=*/{});
}

static std::pair<mlir::LogicalResult, mlir::acc::AtomicUpdateOp>
emitAtomicUpdate(CIRGenFunction &cgf, CIRGenBuilderTy &builder,
                 mlir::Location start, mlir::Location end,
                 const OpenACCAtomicConstruct::SingleStmtInfo &inf) {
  mlir::Value x = cgf.emitLValue(inf.X).getPointer();
  auto op = mlir::acc::AtomicUpdateOp::create(builder, start, x, /*ifCond=*/{});

  mlir::LogicalResult res = mlir::success();
  {
    mlir::OpBuilder::InsertionGuard guardCase(builder);
    mlir::Type argTy = cast<cir::PointerType>(x.getType()).getPointee();
    std::array<mlir::Type, 1> recipeType{argTy};
    std::array<mlir::Location, 1> recipeLoc{start};
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
    mlir::acc::YieldOp::create(builder, end, {load});
  }

  return {res, op};
}

mlir::LogicalResult
CIRGenFunction::emitOpenACCAtomicConstruct(const OpenACCAtomicConstruct &s) {
  // While Atomic is an 'associated statement' construct, it 'steals' the
  // expression it is associated with rather than emitting it inside of it.  So
  // it has custom emit logic.
  mlir::Location start = getLoc(s.getSourceRange().getBegin());
  mlir::Location end = getLoc(s.getSourceRange().getEnd());
  OpenACCAtomicConstruct::StmtInfo inf = s.getAssociatedStmtInfo();

  switch (s.getAtomicKind()) {
  case OpenACCAtomicKind::Read: {
    assert(inf.Form == OpenACCAtomicConstruct::StmtInfo::StmtForm::Read);
    mlir::acc::AtomicReadOp op =
        emitAtomicRead(*this, builder, start, inf.First);
    emitOpenACCClauses(op, s.getDirectiveKind(), s.getDirectiveLoc(),
                       s.clauses());
    return mlir::success();
  }
  case OpenACCAtomicKind::Write: {
    assert(inf.Form == OpenACCAtomicConstruct::StmtInfo::StmtForm::Write);
    auto op = emitAtomicWrite(*this, builder, start, inf.First);
    emitOpenACCClauses(op, s.getDirectiveKind(), s.getDirectiveLoc(),
                       s.clauses());
    return mlir::success();
  }
  case OpenACCAtomicKind::None:
  case OpenACCAtomicKind::Update: {
    assert(inf.Form == OpenACCAtomicConstruct::StmtInfo::StmtForm::Update);
    auto [res, op] = emitAtomicUpdate(*this, builder, start, end, inf.First);
    emitOpenACCClauses(op, s.getDirectiveKind(), s.getDirectiveLoc(),
                       s.clauses());
    return res;
  }
  case OpenACCAtomicKind::Capture: {
    // Atomic-capture is made up of two statements, either an update = read,
    // read + update, or read + write.  As a result, the IR represents the
    // capture region as having those two 'inside' of it.
    auto op = mlir::acc::AtomicCaptureOp::create(builder, start, /*ifCond=*/{});
    emitOpenACCClauses(op, s.getDirectiveKind(), s.getDirectiveLoc(),
                       s.clauses());
    mlir::LogicalResult res = mlir::success();
    {
      mlir::OpBuilder::InsertionGuard guardCase(builder);

      mlir::Block *block =
          builder.createBlock(&op.getRegion(), op.getRegion().end(), {}, {});

      builder.setInsertionPointToStart(block);

      auto terminator = mlir::acc::TerminatorOp::create(builder, end);

      // The AtomicCaptureOp only permits the two acc.atomic.* operations inside
      // of it, so all other parts of the expression need to be emitted before
      // the AtomicCaptureOp, then moved into place.
      builder.setInsertionPoint(op);

      switch (inf.Form) {
      default:
        llvm_unreachable("invalid form for Capture");
      case OpenACCAtomicConstruct::StmtInfo::StmtForm::ReadWrite: {
        mlir::acc::AtomicReadOp first =
            emitAtomicRead(*this, builder, start, inf.First);
        mlir::acc::AtomicWriteOp second =
            emitAtomicWrite(*this, builder, start, inf.Second);

        first->moveBefore(terminator);
        second->moveBefore(terminator);
        break;
      }
      case OpenACCAtomicConstruct::StmtInfo::StmtForm::ReadUpdate: {
        mlir::acc::AtomicReadOp first =
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
        mlir::acc::AtomicReadOp second =
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
