//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emit OpenMP Stmt nodes as CIR code.
//
//===----------------------------------------------------------------------===//

#include "CIRGenBuilder.h"
#include "CIRGenFunction.h"
#include "CIRGenOpenMPClause.h"
#include "CIRGenOpenMPConstructDecomposition.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "clang/AST/OpenMPClause.h"
#include "clang/AST/StmtOpenMP.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Frontend/OpenMP/OMP.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
using namespace clang;
using namespace clang::CIRGen;

mlir::LogicalResult
CIRGenFunction::emitOMPScopeDirective(const OMPScopeDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPScopeDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPErrorDirective(const OMPErrorDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPErrorDirective");
  return mlir::failure();
}

/// Report \p item's synthesized clauses as not-yet-implemented: having no AST
/// node, they cannot go through the clause emitters.
static mlir::LogicalResult
checkSynthesizedClauses(CIRGenFunction &cgf, const OMPExecutableDirective &s,
                        omp::ConstructQueue::const_iterator item) {
  mlir::LogicalResult res = mlir::success();
  for (llvm::omp::Clause synth : item->synthesized) {
    cgf.getCIRGenModule().errorNYI(s.getSourceRange(),
                                   (llvm::Twine("OpenMP synthesized '") +
                                    llvm::omp::getOpenMPClauseName(synth) +
                                    "' clause from construct decomposition")
                                       .str());
    res = mlir::failure();
  }
  return res;
}

static mlir::LogicalResult
emitParallelClauses(CIRGenFunction &cgf, CIRGenModule &cgm,
                    CIRGenBuilderTy &builder, mlir::Location loc,
                    llvm::ArrayRef<const OMPClause *> clauses,
                    mlir::omp::ParallelOperands &clauseOps) {
  OpenMPClauseEmitter ce(cgf, cgm, builder, loc, clauses);
  ce.emitIf(clauseOps, llvm::omp::Directive::OMPD_parallel);
  ce.emitNumThreads(clauseOps);
  ce.emitProcBind(clauseOps);
  return ce.emitNYI</*supported=*/OMPIfClause, OMPNumThreadsClause,
                    OMPProcBindClause>(
      /*nyi=*/OpenMPNYIClauseList<OMPAllocateClause, OMPCopyinClause,
                                  OMPDefaultClause, OMPFirstprivateClause,
                                  OMPPrivateClause, OMPReductionClause,
                                  OMPSharedClause>{},
      llvm::omp::Directive::OMPD_parallel);
}

template <typename DirectiveTy>
static mlir::LogicalResult
emitParallelOp(CIRGenFunction &cgf, const DirectiveTy &s,
               const omp::ConstructQueue &queue,
               omp::ConstructQueue::const_iterator item, mlir::Location begin,
               mlir::Location end, const mlir::omp::ParallelOperands &clauseOps,
               llvm::function_ref<mlir::LogicalResult()> emitBody) {
  CIRGenBuilderTy &builder = cgf.getBuilder();
  CIRGenModule &cgm = cgf.getCIRGenModule();

  auto parallelOp = mlir::omp::ParallelOp::create(builder, begin, clauseOps);
  if (!omp::isLastItemInQueue(item, queue))
    parallelOp.setCombined(true);

  mlir::Block &block = parallelOp.getRegion().emplaceBlock();
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(&block);

  CIRGenFunction::LexicalScope ls{cgf, begin, builder.getInsertionBlock()};

  // hasCancel() reports a nested cancel/cancellation point directive in the
  // body, not a clause on `parallel`, so it can't be an emitParallelClauses()
  // NYI check.
  if (s.hasCancel()) {
    cgm.errorNYI(s.getBeginLoc(), "OpenMP Parallel with Cancel");
    return mlir::failure();
  }
  // Only set for reduction(task: ...), already rejected as NYI above.
  assert(!s.getTaskReductionRefExpr() &&
         "reduction(task: ...) should already be rejected as NYI");

  mlir::LogicalResult res = emitBody();
  mlir::omp::TerminatorOp::create(builder, end);
  return res;
}

mlir::LogicalResult
CIRGenFunction::emitOMPParallelDirective(const OMPParallelDirective &s) {
  mlir::Location begin = getLoc(s.getBeginLoc());
  mlir::Location end = getLoc(s.getEndLoc());

  omp::ConstructQueue queue =
      omp::buildConstructQueue(getContext().getLangOpts().OpenMP, s);
  omp::ConstructQueue::const_iterator item = queue.begin();

  if (mlir::failed(checkSynthesizedClauses(*this, s, item)))
    return mlir::failure();

  mlir::omp::ParallelOperands clauseOps;
  if (mlir::failed(emitParallelClauses(*this, getCIRGenModule(), builder, begin,
                                       item->clauses, clauseOps)))
    return mlir::failure();

  return emitParallelOp(
      *this, s, queue, item, begin, end, clauseOps,
      [&]() -> mlir::LogicalResult {
        // emitStmt() rejects CapturedStmt directly; the parent construct
        // must unwrap it, so emit the inner statement instead.
        const CapturedStmt *cs = s.getCapturedStmt(llvm::omp::OMPD_parallel);
        return emitStmt(cs->getCapturedStmt(), /*useCurrentScope=*/true);
      });
}

/// Converts a CIR integer value to the equivalent builtin MLIR integer type.
static mlir::Value cirIntToBuiltinInt(CIRGenBuilderTy &builder,
                                      mlir::Location loc,
                                      mlir::Value cirValue) {
  auto cirIntType = mlir::cast<cir::IntType>(cirValue.getType());
  mlir::Type builtinIntType = builder.getIntegerType(cirIntType.getWidth());
  return builder.createBuiltinIntCast(loc, cirValue, builtinIntType);
}

/// Emits the Sema-generated pre-init statements for an OpenMP loop directive.
static mlir::LogicalResult emitPreinits(CIRGenFunction &cgf,
                                        const Stmt *preInits) {
  if (!preInits)
    return mlir::success();

  llvm::SmallVector<const Stmt *> stmts;
  if (const auto *compound = dyn_cast<CompoundStmt>(preInits))
    llvm::append_range(stmts, compound->body());
  else
    stmts.push_back(preInits);

  for (const Stmt *stmt : stmts) {
    if (const auto *declStmt = dyn_cast<DeclStmt>(stmt)) {
      for (const Decl *d : declStmt->decls())
        cgf.emitVarDecl(cast<VarDecl>(*d));
    } else {
      if (cgf.emitStmt(stmt, /*useCurrentScope=*/true).failed())
        return mlir::failure();
    }
  }
  return mlir::success();
}

/// Emits an omp.loop_nest that iterates a normalized `[0, tripCount)` range
/// and recomputes the real induction variable each iteration using the given
/// update expression.
static mlir::LogicalResult
emitOMPLoopNest(CIRGenFunction &cgf, const ForStmt &forStmt, mlir::Value lb,
                mlir::Value ub, mlir::Value step, const VarDecl *ivDecl,
                const Expr *update) {
  CIRGenBuilderTy &builder = cgf.getBuilder();
  mlir::Location loc = cgf.getLoc(forStmt.getSourceRange());

  auto loopNestOp = mlir::omp::LoopNestOp::create(
      builder, loc, /*collapse_num_loops=*/1, lb, ub, step,
      /*loop_inclusive=*/false, /*tile_sizes=*/nullptr);
  mlir::Block *block = new mlir::Block();
  loopNestOp.getRegion().push_back(block);
  block->addArgument(lb.getType(), loc);

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(block);

  // Store the normalized counter block argument into the normalized loop
  // variable's storage location, then let Sema's update expression recompute
  // the real induction variable from it.
  mlir::Value iv = block->getArgument(0);
  Address ivAddr = cgf.getAddrOfLocalVar(ivDecl);
  mlir::Value civVal =
      builder.createBuiltinIntCast(loc, iv, ivAddr.getElementType());
  builder.createStore(loc, civVal, ivAddr);
  cgf.emitIgnoredExpr(update);

  mlir::LogicalResult bodyRes = mlir::success();
  if (forStmt.getBody())
    if (cgf.emitStmt(forStmt.getBody(), /*useCurrentScope=*/true).failed())
      bodyRes = mlir::failure();

  mlir::omp::YieldOp::create(builder, cgf.getLoc(forStmt.getEndLoc()));
  return bodyRes;
}

/// Evaluates the clauses allowed on an omp.wsloop leaf; none are supported
/// yet, so every eligible clause is reported as NYI.
static mlir::LogicalResult
emitWsloopClauses(CIRGenFunction &cgf, CIRGenModule &cgm,
                  CIRGenBuilderTy &builder, mlir::Location loc,
                  llvm::ArrayRef<const OMPClause *> clauses,
                  mlir::omp::WsloopOperands &clauseOps) {
  OpenMPClauseEmitter ce(cgf, cgm, builder, loc, clauses);
  return ce.emitNYI</*supported=*/>(
      /*nyi=*/OpenMPNYIClauseList<
          OMPAllocateClause, OMPCollapseClause, OMPFirstprivateClause,
          OMPLastprivateClause, OMPLinearClause, OMPNowaitClause,
          OMPOrderClause, OMPOrderedClause, OMPPrivateClause,
          OMPReductionClause, OMPScheduleClause>{},
      llvm::omp::Directive::OMPD_for);
}

/// Creates a module-level `omp.private` op (a "privatizer") for a scalar
/// of CIR type elemTy.
static mlir::omp::PrivateClauseOp
createCounterPrivatizer(CIRGenModule &cgm, mlir::Location loc,
                        llvm::StringRef baseName, mlir::Type elemTy) {
  mlir::ModuleOp mod = cgm.getModule();
  std::string name = (baseName + ".privatizer").str();
  for (unsigned suffix = 0; mlir::SymbolTable::lookupSymbolIn(mod, name);
       ++suffix)
    name = (baseName + ".privatizer." + llvm::Twine(suffix)).str();

  mlir::OpBuilder b(mod.getContext());
  b.setInsertionPointToEnd(mod.getBody());
  return mlir::omp::PrivateClauseOp::create(b, loc, mlir::TypeRange{},
                                            b.getStringAttr(name),
                                            mlir::TypeAttr::get(elemTy));
}

/// Registers a loop counter as predetermined-private on the omp.wsloop op.
static Address addPrivateCounter(CIRGenFunction &cgf, CIRGenModule &cgm,
                                 mlir::Location loc, const VarDecl *privateVd,
                                 mlir::omp::WsloopOperands &clauseOps) {
  cgf.emitVarDecl(*privateVd);
  Address moldAddr = cgf.getAddrOfLocalVar(privateVd);

  mlir::omp::PrivateClauseOp privatizer = createCounterPrivatizer(
      cgm, loc, privateVd->getName(), moldAddr.getElementType());

  clauseOps.privateVars.push_back(moldAddr.getPointer());
  clauseOps.privateSyms.push_back(
      mlir::FlatSymbolRefAttr::get(privatizer.getSymNameAttr()));
  return moldAddr;
}

/// Lowers an OMPLoopDirective's `for` leaf to an omp.wsloop + omp.loop_nest.
static mlir::LogicalResult
emitOMPWorksharingLoop(CIRGenFunction &cgf, const OMPLoopDirective &s,
                       omp::ConstructQueue::const_iterator item) {
  CIRGenBuilderTy &builder = cgf.getBuilder();
  CIRGenModule &cgm = cgf.getCIRGenModule();
  mlir::Location loc = cgf.getLoc(s.getBeginLoc());

  if (mlir::failed(checkSynthesizedClauses(cgf, s, item)))
    return mlir::failure();

  mlir::omp::WsloopOperands clauseOps;
  if (emitWsloopClauses(cgf, cgm, builder, loc, item->clauses, clauseOps)
          .failed())
    return mlir::failure();

  const CapturedStmt *capturedStmt = s.getInnermostCapturedStmt();
  const auto *forStmt = cast<ForStmt>(capturedStmt->getCapturedStmt());

  if (emitPreinits(cgf, s.getPreInits()).failed())
    return mlir::failure();

  // Allocate storage for Sema's normalized 0-based loop counter.
  const auto *ivDecl =
      cast<VarDecl>(cast<DeclRefExpr>(s.getIterationVariable())->getDecl());
  cgf.emitVarDecl(*ivDecl);
  llvm::SmallVector<CIRGenFunction::DeclMapRevertingRAII, 4> counterShadows;
  llvm::SmallVector<std::pair<const VarDecl *, Address>, 4> counterMolds;
  counterShadows.reserve(s.counters().size());
  counterMolds.reserve(s.counters().size());
  for (auto [counter, privateCounter] :
       llvm::zip_equal(s.counters(), s.private_counters())) {
    const auto *vd = cast<VarDecl>(cast<DeclRefExpr>(counter)->getDecl());
    const auto *privateVd =
        cast<VarDecl>(cast<DeclRefExpr>(privateCounter)->getDecl());
    counterShadows.emplace_back(cgf, vd);
    counterMolds.emplace_back(
        vd, addPrivateCounter(cgf, cgm, loc, privateVd, clauseOps));
  }

  mlir::Value tripCountCir = cgf.emitScalarExpr(s.getNumIterations());
  auto cirIntType = mlir::cast<cir::IntType>(tripCountCir.getType());
  mlir::Value zero =
      cirIntToBuiltinInt(builder, loc, builder.getConstInt(loc, cirIntType, 0));
  mlir::Value one =
      cirIntToBuiltinInt(builder, loc, builder.getConstInt(loc, cirIntType, 1));
  mlir::Value tripCount = cirIntToBuiltinInt(builder, loc, tripCountCir);

  auto wsloopOp = mlir::omp::WsloopOp::create(builder, loc, clauseOps);
  mlir::Block *innerBlock = new mlir::Block();
  // Ensures the block arguments match BlockArgOpenMPOpInterface's expected
  // order.
  for (auto &counterMold : counterMolds)
    innerBlock->addArgument(counterMold.second.getPointer().getType(), loc);
  wsloopOp.getRegion().push_back(innerBlock);

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(innerBlock);

  // Bind each counter to its private block argument.
  for (unsigned idx = 0; idx < counterMolds.size(); ++idx) {
    const VarDecl *vd = counterMolds[idx].first;
    Address moldAddr = counterMolds[idx].second;
    Address privAddr(innerBlock->getArgument(idx), moldAddr.getElementType(),
                     moldAddr.getAlignment());
    cgf.replaceAddrOfLocalVar(vd, privAddr);
    cgf.symbolTable.insert(vd, privAddr.getPointer());
  }

  return emitOMPLoopNest(cgf, *forStmt, zero, tripCount, one, ivDecl,
                         s.updates()[0]);
}

mlir::LogicalResult
CIRGenFunction::emitOMPTaskwaitDirective(const OMPTaskwaitDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPTaskwaitDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTaskyieldDirective(const OMPTaskyieldDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTaskyieldDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPBarrierDirective(const OMPBarrierDirective &s) {
  mlir::omp::BarrierOp::create(builder, getLoc(s.getBeginLoc()));
  assert(s.clauses().empty() && "omp barrier doesn't support clauses");
  return mlir::success();
}
mlir::LogicalResult
CIRGenFunction::emitOMPMetaDirective(const OMPMetaDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPMetaDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPCanonicalLoop(const OMPCanonicalLoop &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPCanonicalLoop");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPSimdDirective(const OMPSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTileDirective(const OMPTileDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPTileDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPUnrollDirective(const OMPUnrollDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPUnrollDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPFuseDirective(const OMPFuseDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPFuseDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPForDirective(const OMPForDirective &s) {
  omp::ConstructQueue queue =
      omp::buildConstructQueue(getContext().getLangOpts().OpenMP, s);
  return emitOMPWorksharingLoop(*this, s, queue.begin());
}
mlir::LogicalResult
CIRGenFunction::emitOMPForSimdDirective(const OMPForSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPForSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPSectionsDirective(const OMPSectionsDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPSectionsDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPSectionDirective(const OMPSectionDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPSectionDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPSingleDirective(const OMPSingleDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPSingleDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPMasterDirective(const OMPMasterDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPMasterDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPCriticalDirective(const OMPCriticalDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPCriticalDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPParallelForDirective(const OMPParallelForDirective &s) {
  mlir::Location begin = getLoc(s.getBeginLoc());
  mlir::Location end = getLoc(s.getEndLoc());

  omp::ConstructQueue queue =
      omp::buildConstructQueue(getContext().getLangOpts().OpenMP, s);
  omp::ConstructQueue::const_iterator parallelItem = queue.begin();
  assert(parallelItem->id == llvm::omp::OMPD_parallel &&
         "expected 'parallel' to be the outermost leaf");

  if (mlir::failed(checkSynthesizedClauses(*this, s, parallelItem)))
    return mlir::failure();

  mlir::omp::ParallelOperands parallelOps;
  if (mlir::failed(emitParallelClauses(*this, getCIRGenModule(), builder, begin,
                                       parallelItem->clauses, parallelOps)))
    return mlir::failure();

  return emitParallelOp(
      *this, s, queue, parallelItem, begin, end, parallelOps,
      [&]() -> mlir::LogicalResult {
        omp::ConstructQueue::const_iterator forItem = std::next(parallelItem);
        assert(forItem != queue.end() && forItem->id == llvm::omp::OMPD_for &&
               "expected a 'for' leaf nested in 'parallel'");
        return emitOMPWorksharingLoop(*this, s, forItem);
      });
}
mlir::LogicalResult CIRGenFunction::emitOMPParallelForSimdDirective(
    const OMPParallelForSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPParallelForSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPParallelMasterDirective(
    const OMPParallelMasterDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPParallelMasterDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPParallelSectionsDirective(
    const OMPParallelSectionsDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPParallelSectionsDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTaskDirective(const OMPTaskDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPTaskDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTaskgroupDirective(const OMPTaskgroupDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTaskgroupDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPFlushDirective(const OMPFlushDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPFlushDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPDepobjDirective(const OMPDepobjDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPDepobjDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPScanDirective(const OMPScanDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPScanDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPOrderedStandaloneDirective(
    const OMPOrderedStandaloneDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPOrderedStandaloneDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPOrderedBlockAssocDirective(
    const OMPOrderedBlockAssocDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPOrderedBlockAssocDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPAtomicDirective(const OMPAtomicDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPAtomicDirective");
  return mlir::failure();
}

/// Check for unsupported implicit captures in a target region.
static mlir::LogicalResult
emitOMPTargetImplicitCaptures(CIRGenFunction &cgf,
                              const OMPExecutableDirective &s,
                              llvm::ArrayRef<const VarDecl *> mapSyms) {
  const CapturedStmt *cs = s.getCapturedStmt(llvm::omp::OMPD_target);
  mlir::LogicalResult res = mlir::success();
  for (const auto &capture : cs->captures()) {
    if (capture.capturesThis()) {
      cgf.getCIRGenModule().errorNYI(s.getBeginLoc(),
                                     "OpenMP target capture of 'this' pointer");
      res = mlir::failure();
      continue;
    }
    if (capture.capturesVariableByCopy()) {
      cgf.getCIRGenModule().errorNYI(s.getBeginLoc(),
                                     "OpenMP target capture by copy");
      res = mlir::failure();
      continue;
    }
    if (capture.capturesVariableArrayType()) {
      cgf.getCIRGenModule().errorNYI(
          s.getBeginLoc(),
          "OpenMP target capture of variable-length array type");
      res = mlir::failure();
      continue;
    }
    if (capture.capturesVariable()) {
      const VarDecl *vd = capture.getCapturedVar();
      if (llvm::is_contained(mapSyms, vd))
        continue;

      cgf.getCIRGenModule().errorNYI(s.getBeginLoc(),
                                     "OpenMP target implicit by-ref capture");
      res = mlir::failure();
    }
  }
  return res;
}

static mlir::LogicalResult
emitTargetClauses(CIRGenFunction &cgf, CIRGenModule &cgm,
                  CIRGenBuilderTy &builder, mlir::Location loc,
                  llvm::ArrayRef<const OMPClause *> clauses,
                  mlir::omp::TargetExtOperands &clauseOps,
                  llvm::SmallVectorImpl<const VarDecl *> &mapSyms) {
  OpenMPClauseEmitter ce(cgf, cgm, builder, loc, clauses);
  ce.emitMap(clauseOps, &mapSyms);
  return ce.emitNYI</*supported=*/OMPMapClause>(
      /*nyi=*/OpenMPNYIClauseList<
          OMPAllocateClause, OMPDefaultClause, OMPDefaultmapClause,
          OMPDependClause, OMPDeviceClause, OMPFirstprivateClause,
          OMPHasDeviceAddrClause, OMPIfClause, OMPInReductionClause,
          OMPIsDevicePtrClause, OMPNowaitClause, OMPPrivateClause,
          OMPThreadLimitClause, OMPUsesAllocatorsClause, OMPXBareClause>{},
      llvm::omp::Directive::OMPD_target);
}

template <typename DirectiveTy>
static mlir::LogicalResult
emitTargetOp(CIRGenFunction &cgf, const DirectiveTy &s,
             const omp::ConstructQueue &queue,
             omp::ConstructQueue::const_iterator item, mlir::Location begin,
             mlir::Location end, mlir::omp::TargetExtOperands &clauseOps,
             llvm::ArrayRef<const VarDecl *> mapSyms,
             llvm::function_ref<mlir::LogicalResult()> emitBody) {
  CIRGenBuilderTy &builder = cgf.getBuilder();

  if (mlir::failed(emitOMPTargetImplicitCaptures(cgf, s, mapSyms)))
    return mlir::failure();

  // Use generic for now.
  clauseOps.kernelType = mlir::omp::TargetExecModeAttr::get(
      &cgf.getMLIRContext(), mlir::omp::TargetExecMode::generic);

  auto targetOp = mlir::omp::TargetOp::create(builder, begin, clauseOps);
  if (!omp::isLastItemInQueue(item, queue))
    targetOp.setCombined(true);

  mlir::Block &block = targetOp.getRegion().emplaceBlock();
  for (mlir::Value mapVar : clauseOps.mapVars)
    block.addArgument(mapVar.getType(), begin);

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(&block);

  CIRGenFunction::LexicalScope ls{cgf, begin, builder.getInsertionBlock()};

  llvm::SmallVector<std::pair<const VarDecl *, Address>> savedAddrs;
  for (auto [idx, vd] : llvm::enumerate(mapSyms)) {
    Address origAddr = cgf.getAddrOfLocalVar(vd);
    savedAddrs.push_back({vd, origAddr});
    mlir::Value blockArg = block.getArgument(idx);
    cgf.replaceAddrOfLocalVar(vd, Address(blockArg, origAddr.getAlignment()));
  }

  mlir::LogicalResult res = emitBody();
  mlir::omp::TerminatorOp::create(builder, end);

  for (auto &[vd, addr] : savedAddrs)
    cgf.replaceAddrOfLocalVar(vd, addr);

  return res;
}

mlir::LogicalResult
CIRGenFunction::emitOMPTargetDirective(const OMPTargetDirective &s) {
  mlir::Location begin = getLoc(s.getBeginLoc());
  mlir::Location end = getLoc(s.getEndLoc());

  omp::ConstructQueue queue =
      omp::buildConstructQueue(getContext().getLangOpts().OpenMP, s);
  omp::ConstructQueue::const_iterator item = queue.begin();

  if (mlir::failed(checkSynthesizedClauses(*this, s, item)))
    return mlir::failure();

  mlir::omp::TargetExtOperands clauseOps;
  llvm::SmallVector<const VarDecl *> mapSyms;
  if (mlir::failed(emitTargetClauses(*this, getCIRGenModule(), builder, begin,
                                     item->clauses, clauseOps, mapSyms)))
    return mlir::failure();

  return emitTargetOp(*this, s, queue, item, begin, end, clauseOps, mapSyms,
                      [&]() -> mlir::LogicalResult {
                        const CapturedStmt *cs =
                            s.getCapturedStmt(llvm::omp::OMPD_target);
                        return emitStmt(cs->getCapturedStmt(),
                                        /*useCurrentScope=*/true);
                      });
}
mlir::LogicalResult
CIRGenFunction::emitOMPTeamsDirective(const OMPTeamsDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPTeamsDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPCancellationPointDirective(
    const OMPCancellationPointDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPCancellationPointDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPCancelDirective(const OMPCancelDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPCancelDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTargetDataDirective(const OMPTargetDataDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetDataDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetEnterDataDirective(
    const OMPTargetEnterDataDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetEnterDataDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetExitDataDirective(
    const OMPTargetExitDataDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetExitDataDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetParallelDirective(
    const OMPTargetParallelDirective &s) {
  mlir::Location begin = getLoc(s.getBeginLoc());
  mlir::Location end = getLoc(s.getEndLoc());

  omp::ConstructQueue queue =
      omp::buildConstructQueue(getContext().getLangOpts().OpenMP, s);
  omp::ConstructQueue::const_iterator targetItem = queue.begin();
  assert(targetItem->id == llvm::omp::OMPD_target &&
         "expected 'target' to be the outermost leaf");

  if (mlir::failed(checkSynthesizedClauses(*this, s, targetItem)))
    return mlir::failure();

  mlir::omp::TargetExtOperands targetOps;
  llvm::SmallVector<const VarDecl *> mapSyms;
  if (mlir::failed(emitTargetClauses(*this, getCIRGenModule(), builder, begin,
                                     targetItem->clauses, targetOps, mapSyms)))
    return mlir::failure();

  return emitTargetOp(
      *this, s, queue, targetItem, begin, end, targetOps, mapSyms,
      [&]() -> mlir::LogicalResult {
        omp::ConstructQueue::const_iterator parallelItem =
            std::next(targetItem);
        assert(parallelItem != queue.end() &&
               parallelItem->id == llvm::omp::OMPD_parallel &&
               "expected a 'parallel' leaf nested in 'target'");

        if (mlir::failed(checkSynthesizedClauses(*this, s, parallelItem)))
          return mlir::failure();

        mlir::omp::ParallelOperands parallelOps;
        if (mlir::failed(emitParallelClauses(*this, getCIRGenModule(), builder,
                                             begin, parallelItem->clauses,
                                             parallelOps)))
          return mlir::failure();

        return emitParallelOp(*this, s, queue, parallelItem, begin, end,
                              parallelOps, [&]() -> mlir::LogicalResult {
                                const CapturedStmt *cs =
                                    s.getCapturedStmt(llvm::omp::OMPD_parallel);
                                return emitStmt(cs->getCapturedStmt(),
                                                /*useCurrentScope=*/true);
                              });
      });
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetParallelForDirective(
    const OMPTargetParallelForDirective &s) {
  mlir::Location begin = getLoc(s.getBeginLoc());
  mlir::Location end = getLoc(s.getEndLoc());

  omp::ConstructQueue queue =
      omp::buildConstructQueue(getContext().getLangOpts().OpenMP, s);
  omp::ConstructQueue::const_iterator targetItem = queue.begin();
  assert(targetItem->id == llvm::omp::OMPD_target &&
         "expected 'target' to be the outermost leaf");

  if (mlir::failed(checkSynthesizedClauses(*this, s, targetItem)))
    return mlir::failure();

  mlir::omp::TargetExtOperands targetOps;
  llvm::SmallVector<const VarDecl *> mapSyms;
  if (mlir::failed(emitTargetClauses(*this, getCIRGenModule(), builder, begin,
                                     targetItem->clauses, targetOps, mapSyms)))
    return mlir::failure();

  return emitTargetOp(
      *this, s, queue, targetItem, begin, end, targetOps, mapSyms,
      [&]() -> mlir::LogicalResult {
        omp::ConstructQueue::const_iterator parallelItem =
            std::next(targetItem);
        assert(parallelItem != queue.end() &&
               parallelItem->id == llvm::omp::OMPD_parallel &&
               "expected a 'parallel' leaf nested in 'target'");

        if (mlir::failed(checkSynthesizedClauses(*this, s, parallelItem)))
          return mlir::failure();

        mlir::omp::ParallelOperands parallelOps;
        if (mlir::failed(emitParallelClauses(*this, getCIRGenModule(), builder,
                                             begin, parallelItem->clauses,
                                             parallelOps)))
          return mlir::failure();

        return emitParallelOp(
            *this, s, queue, parallelItem, begin, end, parallelOps,
            [&]() -> mlir::LogicalResult {
              omp::ConstructQueue::const_iterator forItem =
                  std::next(parallelItem);
              assert(forItem != queue.end() &&
                     forItem->id == llvm::omp::OMPD_for &&
                     "expected a 'for' leaf nested in 'parallel'");
              return emitOMPWorksharingLoop(*this, s, forItem);
            });
      });
}
mlir::LogicalResult
CIRGenFunction::emitOMPTaskLoopDirective(const OMPTaskLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPTaskLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTaskLoopSimdDirective(
    const OMPTaskLoopSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTaskLoopSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPMaskedTaskLoopDirective(
    const OMPMaskedTaskLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPMaskedTaskLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPMaskedTaskLoopSimdDirective(
    const OMPMaskedTaskLoopSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPMaskedTaskLoopSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPMasterTaskLoopDirective(
    const OMPMasterTaskLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPMasterTaskLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPMasterTaskLoopSimdDirective(
    const OMPMasterTaskLoopSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPMasterTaskLoopSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPParallelGenericLoopDirective(
    const OMPParallelGenericLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPParallelGenericLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPParallelMaskedDirective(
    const OMPParallelMaskedDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPParallelMaskedDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPParallelMaskedTaskLoopDirective(
    const OMPParallelMaskedTaskLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPParallelMaskedTaskLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPParallelMaskedTaskLoopSimdDirective(
    const OMPParallelMaskedTaskLoopSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPParallelMaskedTaskLoopSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPParallelMasterTaskLoopDirective(
    const OMPParallelMasterTaskLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPParallelMasterTaskLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPParallelMasterTaskLoopSimdDirective(
    const OMPParallelMasterTaskLoopSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPParallelMasterTaskLoopSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPDistributeDirective(const OMPDistributeDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPDistributeDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPDistributeParallelForDirective(
    const OMPDistributeParallelForDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPDistributeParallelForDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPDistributeParallelForSimdDirective(
    const OMPDistributeParallelForSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPDistributeParallelForSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPDistributeSimdDirective(
    const OMPDistributeSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPDistributeSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetParallelGenericLoopDirective(
    const OMPTargetParallelGenericLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetParallelGenericLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetParallelForSimdDirective(
    const OMPTargetParallelForSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetParallelForSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTargetSimdDirective(const OMPTargetSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetTeamsGenericLoopDirective(
    const OMPTargetTeamsGenericLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetTeamsGenericLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetUpdateDirective(
    const OMPTargetUpdateDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetUpdateDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTeamsDistributeDirective(
    const OMPTeamsDistributeDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTeamsDistributeDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTeamsDistributeSimdDirective(
    const OMPTeamsDistributeSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTeamsDistributeSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTeamsDistributeParallelForSimdDirective(
    const OMPTeamsDistributeParallelForSimdDirective &s) {
  getCIRGenModule().errorNYI(
      s.getSourceRange(), "OpenMP OMPTeamsDistributeParallelForSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTeamsDistributeParallelForDirective(
    const OMPTeamsDistributeParallelForDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTeamsDistributeParallelForDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTeamsGenericLoopDirective(
    const OMPTeamsGenericLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTeamsGenericLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTargetTeamsDirective(const OMPTargetTeamsDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetTeamsDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetTeamsDistributeDirective(
    const OMPTargetTeamsDistributeDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetTeamsDistributeDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTargetTeamsDistributeParallelForDirective(
    const OMPTargetTeamsDistributeParallelForDirective &s) {
  getCIRGenModule().errorNYI(
      s.getSourceRange(),
      "OpenMP OMPTargetTeamsDistributeParallelForDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTargetTeamsDistributeParallelForSimdDirective(
    const OMPTargetTeamsDistributeParallelForSimdDirective &s) {
  getCIRGenModule().errorNYI(
      s.getSourceRange(),
      "OpenMP OMPTargetTeamsDistributeParallelForSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetTeamsDistributeSimdDirective(
    const OMPTargetTeamsDistributeSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetTeamsDistributeSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPInteropDirective(const OMPInteropDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPInteropDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPDispatchDirective(const OMPDispatchDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPDispatchDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPGenericLoopDirective(const OMPGenericLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPGenericLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPReverseDirective(const OMPReverseDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPReverseDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPSplitDirective(const OMPSplitDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPSplitDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPInterchangeDirective(const OMPInterchangeDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPInterchangeDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPFlattenDirective(const OMPFlattenDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPFlattenDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPAssumeDirective(const OMPAssumeDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPAssumeDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPMaskedDirective(const OMPMaskedDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPMaskedDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPStripeDirective(const OMPStripeDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPStripeDirective");
  return mlir::failure();
}
