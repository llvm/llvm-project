#include "clang/CIR/Sema/FallThroughWarning.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace cir;
using namespace clang;

namespace clang {

//===----------------------------------------------------------------------===//
// Check for missing return value.
//===----------------------------------------------------------------------===//

bool CheckFallThroughDiagnostics::checkDiagnostics(DiagnosticsEngine &d,
                                                   bool returnsVoid,
                                                   bool hasNoReturn) const {
  if (funKind == diag::FalloffFunctionKind::Function) {
    return (returnsVoid || d.isIgnored(diag::warn_falloff_nonvoid, funcLoc)) &&
           (!hasNoReturn ||
            d.isIgnored(diag::warn_noreturn_has_return_expr, funcLoc)) &&
           (!returnsVoid ||
            d.isIgnored(diag::warn_suggest_noreturn_block, funcLoc));
  }
  if (funKind == diag::FalloffFunctionKind::Coroutine) {
    return (returnsVoid || d.isIgnored(diag::warn_falloff_nonvoid, funcLoc)) &&
           (!hasNoReturn);
  }
  // For blocks / lambdas.
  return returnsVoid && !hasNoReturn;
}

// TODO: Add a class for fall through config later

void FallThroughWarningPass::checkFallThroughForFuncBody(
    Sema &s, cir::FuncOp cfg, QualType blockType,
    const CheckFallThroughDiagnostics &cd) {

  llvm::errs() << "Hello world, you're in CIR sema analysis\n";
  bool returnsVoid = false;
  bool hasNoReturn = false;

  // Supposedly all function in cir is FuncOp
  // 1. If normal function (FunctionDecl), check if it's coroutine.
  // 1a. if coroutine -> check the fallthrough handler (idk what this means,
  // TODO for now)
  if (cfg.getCoroutine()) {
    // TODO: Let's not worry about coroutine for now
  } else
    returnsVoid = isa<cir::VoidType>(cfg.getFunctionType().getReturnType());

  // TODO: Do we need to check for InferredNoReturnAttr just like in OG?
  hasNoReturn = cfg.getFunctionType().getReturnTypes().empty();

  DiagnosticsEngine &diags = s.getDiagnostics();
  if (cd.checkDiagnostics(diags, returnsVoid, hasNoReturn)) {
    return;
  }

  // cpu_dispatch functions permit empty function bodies for ICC compatibility.
  // TODO: Do we have isCPUDispatchMultiVersion?
  checkFallThrough(cfg);
}

mlir::DenseSet<mlir::Block *>
FallThroughWarningPass::getLiveSet(cir::FuncOp cfg) {
  mlir::DenseSet<mlir::Block *> liveSet;
  if (cfg.getBody().empty())
    return liveSet;

  auto &first = cfg.getBody().getBlocks().front();

  for (auto &block : cfg.getBody()) {
    if (first.isReachable(&block))
      liveSet.insert(&block);
  }
  return liveSet;
}

ControlFlowKind FallThroughWarningPass::checkFallThrough(cir::FuncOp cfg) {

  assert(cfg && "there can't be a null func op");

  // TODO: Is no CFG akin to a declaration?
  if (cfg.isDeclaration()) {
    return UnknownFallThrough;
  }

  mlir::DenseSet<mlir::Block *> liveSet = this->getLiveSet(cfg);

  unsigned count = liveSet.size();

  bool hasLiveReturn = false;
  bool hasFakeEdge = false;
  bool hasPlainEdge = false;
  bool hasAbnormalEdge = false;

  auto &exitBlock = cfg.getBody().back();
  // INFO: in OG clang CFG, they have an empty exit block, so when they query
  // pred of exit OG, they get all exit blocks
  //
  // I guess in CIR, we can pretend exit blocks are all blocks that have no
  // successor?
  for (mlir::Block &pred : cfg.getBody().getBlocks()) {
    if (!liveSet.contains(&pred))
      continue;

    // We consider no predecessors as 'exit blocks'
    if (!pred.hasNoSuccessors())
      continue;

    if (!pred.mightHaveTerminator())
      continue;

    mlir::Operation *term = pred.getTerminator();
    if (isa<cir::ReturnOp>(term)) {
      hasAbnormalEdge = true;
      continue;
    }

    // INFO: In OG, we'll be looking for destructor since it can appear past
    // return but i guess not in CIR? In this case we'll only be examining the
    // terminator

    if (isa<cir::TryOp>(term)) {
      hasAbnormalEdge = true;
      continue;
    }

    // INFO: OG clang has this equals true whenever ri == re, which means this
    // is true only when a block only has the terminator, or its size is 1.
    hasPlainEdge = std::distance(pred.begin(), pred.end()) == 1;

    if (isa<cir::ReturnOp>(term)) {
      hasLiveReturn = true;
      continue;
    }
    if (isa<cir::TryOp>(term)) {
      hasLiveReturn = true;
      continue;
    }

    // TODO: Maybe one day throw will be terminator?
    //
    // TODO: We need to add a microsoft inline assembly enum

    // TODO: We don't concer with try op either since it's not terminator

    hasPlainEdge = true;
  }

  if (!hasPlainEdge) {
    if (hasLiveReturn)
      return NeverFallThrough;
    return NeverFallThroughOrReturn;
  }
  if (hasAbnormalEdge || hasFakeEdge || hasLiveReturn)
    return MaybeFallThrough;
  // This says AlwaysFallThrough for calls to functions that are not marked
  // noreturn, that don't return.  If people would like this warning to be more
  // accurate, such functions should be marked as noreturn.
  return AlwaysFallThrough;
}
} // namespace clang
