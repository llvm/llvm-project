#include "clang/CIR/Analysis/FallThroughWarning.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "cir-fallthrough"

using namespace mlir;
using namespace cir;
using namespace clang;

namespace clang {

//===----------------------------------------------------------------------===//
// Helper function to lookup a Decl by name from ASTContext
//===----------------------------------------------------------------------===//

/// Lookup a declaration by name in the translation unit.
/// \param Context The ASTContext to search in
/// \param Name The name of the declaration to find
/// \return The found Decl, or nullptr if not found

/// WARN: I have to say, we only use this because a lot of the time, attribute
/// that we might need are not port to CIR currently so this function is
/// basically a crutch for that
Decl *getDeclByName(ASTContext &context, StringRef name) {
  // Get the identifier for the name
  IdentifierInfo *ii = &context.Idents.get(name);

  // Create a DeclarationName from the identifier
  DeclarationName dName(ii);

  // Lookup in the translation unit
  TranslationUnitDecl *tu = context.getTranslationUnitDecl();
  DeclContext::lookup_result result = tu->lookup(dName);

  // Return the first match, or nullptr if not found
  if (result.empty())
    return nullptr;

  return result.front();
}

CheckFallThroughDiagnostics
CheckFallThroughDiagnostics::makeForFunction(Sema &s, const Decl *func) {
  CheckFallThroughDiagnostics d;
  d.funcLoc = func->getLocation();
  d.diagFallThroughHasNoReturn = diag::warn_noreturn_has_return_expr;
  d.diagFallThroughReturnsNonVoid = diag::warn_falloff_nonvoid;

  // Don't suggest that virtual functions be marked "noreturn", since they
  // might be overridden by non-noreturn functions.
  bool isVirtualMethod = false;
  if (const CXXMethodDecl *method = dyn_cast<CXXMethodDecl>(func))
    isVirtualMethod = method->isVirtual();

  // Don't suggest that template instantiations be marked "noreturn"
  bool isTemplateInstantiation = false;
  if (const FunctionDecl *function = dyn_cast<FunctionDecl>(func)) {
    isTemplateInstantiation = function->isTemplateInstantiation();
    if (!s.getLangOpts().CPlusPlus && !s.getLangOpts().C99 &&
        function->isMain()) {
      d.diagFallThroughReturnsNonVoid = diag::ext_main_no_return;
    }
  }

  if (!isVirtualMethod && !isTemplateInstantiation)
    d.diagNeverFallThroughOrReturn = diag::warn_suggest_noreturn_function;

  d.funKind = diag::FalloffFunctionKind::Function;
  return d;
}

CheckFallThroughDiagnostics
CheckFallThroughDiagnostics::makeForCoroutine(const Decl *func) {
  CheckFallThroughDiagnostics d;
  d.funcLoc = func->getLocation();
  d.diagFallThroughReturnsNonVoid = diag::warn_falloff_nonvoid;
  d.funKind = diag::FalloffFunctionKind::Coroutine;
  return d;
}

CheckFallThroughDiagnostics CheckFallThroughDiagnostics::makeForBlock() {
  CheckFallThroughDiagnostics D;
  D.diagFallThroughHasNoReturn = diag::err_noreturn_has_return_expr;
  D.diagFallThroughReturnsNonVoid = diag::err_falloff_nonvoid;
  D.funKind = diag::FalloffFunctionKind::Block;
  return D;
}

CheckFallThroughDiagnostics CheckFallThroughDiagnostics::makeForLambda() {
  CheckFallThroughDiagnostics d;
  d.diagFallThroughHasNoReturn = diag::err_noreturn_has_return_expr;
  d.diagFallThroughReturnsNonVoid = diag::warn_falloff_nonvoid;
  d.funKind = diag::FalloffFunctionKind::Lambda;
  return d;
}

//===----------------------------------------------------------------------===//
// Check for phony return values (returning uninitialized __retval)
//===----------------------------------------------------------------------===//

/// Check if a return operation returns a phony value.
/// A phony return is when a function returns a value loaded from an
/// uninitialized __retval alloca, which indicates the function doesn't
/// actually return a meaningful value.
///
/// Example of phony return:
/// \code
///   cir.func @test1() -> !s32i {
///     %0 = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
///     %1 = cir.load %0 : !cir.ptr<!s32i>, !s32i
///     cir.return %1 : !s32i
///   }
/// \endcode
///
/// \param returnOp The return operation to check
/// \return true if this is a phony return, false otherwise
bool isPhonyReturn(cir::ReturnOp returnOp) {
  assert(returnOp && "ReturnOp should be non-null");

  if (returnOp.getIsImplicit())
    return true;

  // Get the returned value - return operations use $input as the operand
  if (!returnOp.hasOperand())
    return false;

  auto returnValue = returnOp.getInput()[0];

  auto loadOp = returnValue.getDefiningOp<cir::LoadOp>();
  if (!loadOp)
    return false;

  // Check if the load is from an alloca
  auto allocaOp = loadOp.getAddr().getDefiningOp<cir::AllocaOp>();
  if (!allocaOp)
    return false;

  // Check if the alloca is named "__retval"
  auto name = allocaOp.getName();
  if (name != "__retval")
    return false;

  // Check if there are ANY stores to __retval in the entire function.
  // This is intentionally path-INsensitive - if there are stores on some
  // paths, then this return is considered non-phony.
  // The control flow analysis (hasLiveReturn + hasPlainEdge) will determine
  // if all paths return properly.
  mlir::Value allocaResult = allocaOp.getResult();

  for (auto *user : allocaResult.getUsers()) {
    if (auto storeOp = dyn_cast<cir::StoreOp>(user)) {
      if (storeOp.getAddr() == allocaResult) {
        // There's a store to __retval somewhere - not a phony return
        return false;
      }
    }
  }

  // No stores to __retval anywhere - this is a phony return (uninitialized)
  return true;
}

//===----------------------------------------------------------------------===//
// Check for missing return value.
//===----------------------------------------------------------------------===//

bool CheckFallThroughDiagnostics::checkDiagnostics(DiagnosticsEngine &d,
                                                   bool returnsVoid,
                                                   bool hasNoReturn) const {
  if (funKind == diag::FalloffFunctionKind::Function) {
    return (returnsVoid || d.isIgnored(diag::warn_falloff_nonvoid, funcLoc)) &&
           (d.isIgnored(diag::warn_noreturn_has_return_expr, funcLoc) ||
            !hasNoReturn) &&
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

  auto *d = getDeclByName(s.getASTContext(), cfg.getName());
  assert(d && "we need non null decl");
  auto *body = d->getBody();

  // Functions without bodies (declarations only) don't need fall-through
  // analysis
  if (!body)
    return;

  bool returnsVoid = false;
  bool hasNoReturn = false;

  SourceLocation lBrace = body->getBeginLoc(), rBrace = body->getEndLoc();
  // Supposedly all function in cir is FuncOp
  // 1. If normal function (FunctionDecl), check if it's coroutine.
  // 1a. if coroutine -> check the fallthrough handler (idk what this means,
  // TODO for now)
  if (const auto *fd = dyn_cast<FunctionDecl>(d)) {
    if (const auto *cBody = dyn_cast<CoroutineBodyStmt>(d->getBody()))
      returnsVoid = cBody->getFallthroughHandler() != nullptr;
    else
      returnsVoid = fd->getReturnType()->isVoidType();
    hasNoReturn = fd->isNoReturn() || fd->hasAttr<InferredNoReturnAttr>();
  } else if (const auto *md = dyn_cast<ObjCMethodDecl>(d)) {
    returnsVoid = md->getReturnType()->isVoidType();
    hasNoReturn = md->hasAttr<NoReturnAttr>();
  } else if (isa<BlockDecl>(d)) {
    if (const FunctionType *ft =
            blockType->getPointeeType()->getAs<FunctionType>()) {
      if (ft->getReturnType()->isVoidType())
        returnsVoid = true;
      if (ft->getNoReturnAttr())
        hasNoReturn = true;
    }
  }

  DiagnosticsEngine &diags = s.getDiagnostics();

  // Short circuit for compilation speed.
  if (cd.checkDiagnostics(diags, returnsVoid, hasNoReturn))
    return;

  // cpu_dispatch functions permit empty function bodies for ICC compatibility.
  // TODO: Do we have isCPUDispatchMultiVersion?

  switch (ControlFlowKind fallThroughType = checkFallThrough(cfg)) {
  case UnknownFallThrough:
    break;
  case MaybeFallThrough:
    [[fallthrough]];
  case AlwaysFallThrough:
    if (hasNoReturn && cd.diagFallThroughHasNoReturn) {

    } else if (!returnsVoid && cd.diagFallThroughReturnsNonVoid) {
      // If the final statement is a call to an always-throwing function,
      // don't warn about the fall-through.
      if (d->getAsFunction()) {
        if (const auto *cs = dyn_cast<CompoundStmt>(body);
            cs && !cs->body_empty()) {
          const Stmt *lastStmt = cs->body_back();
          // Unwrap ExprWithCleanups if necessary.
          if (const auto *ewc = dyn_cast<ExprWithCleanups>(lastStmt)) {
            lastStmt = ewc->getSubExpr();
          }
          if (const auto *ce = dyn_cast<CallExpr>(lastStmt)) {
            if (const FunctionDecl *callee = ce->getDirectCallee();
                callee && callee->hasAttr<InferredNoReturnAttr>()) {
              return; // Don't warn about fall-through.
            }
          }
          // Direct throw.
          if (isa<CXXThrowExpr>(lastStmt)) {
            return; // Don't warn about fall-through.
          }
        }
      }
      bool notInAllControlPaths = fallThroughType == MaybeFallThrough;
      s.Diag(rBrace, cd.diagFallThroughReturnsNonVoid)
          << cd.funKind << notInAllControlPaths;
    }
    break;
  case NeverFallThroughOrReturn:
    if (returnsVoid && !hasNoReturn && cd.diagNeverFallThroughOrReturn) {
    }
    break;

  case NeverFallThrough: {
  } break;
  }
}

mlir::SetVector<mlir::Block *>
FallThroughWarningPass::getLiveSet(cir::FuncOp cfg) {
  mlir::SetVector<mlir::Block *> liveSet;
  if (cfg.getBody().empty())
    return liveSet;

  DataFlowSolver solver;
  solver.load<mlir::dataflow::DeadCodeAnalysis>();
  int blockCount = 0;
  auto result = solver.initializeAndRun(cfg);
  if (result.failed()) {
    llvm::errs() << "Failure to perform deadcode analysis for ClangIR "
                    "analyzer, returning empty live set\n";
  } else {
    cfg->walk([&](mlir::Block *op) {
      blockCount++;
      if (solver.lookupState<mlir::dataflow::Executable>(
              solver.getProgramPointBefore(op))) {
        liveSet.insert(op);
      }
    });
  }
  LLVM_DEBUG({
    llvm::dbgs() << "=== DCA result for cir analysis fallthrough for "
                 << cfg.getName() << " ===\n";
    llvm::dbgs() << "Live set size: " << liveSet.size() << "\n";
    llvm::dbgs() << "Block traversal count: " << blockCount << "\n";
    llvm::dbgs() << "Dead blocks: " << (blockCount - liveSet.size()) << "\n";
    llvm::dbgs() << "If this is unexpected, please file an issue via GitHub "
                    "with mlir tag"
                 << "\n";
  });

  return liveSet;
}

//===----------------------------------------------------------------------===//
// Switch/Case Analysis Helpers
//===----------------------------------------------------------------------===//

/// Check if a switch operation returns on all code paths. Right now this only
/// works with case default Returns true if:
/// - Switch is in simple form AND
/// - Has a default case that returns
///
/// \param switchOp The switch operation to analyze
/// \return true if all paths through the switch return a value
static bool switchDefaultsWithCoveredEnums(cir::SwitchOp switchOp) {
  llvm::SmallVector<cir::CaseOp> cases;
  if (!switchOp.isSimpleForm(cases))
    return false;

  // Check if there's a default case
  bool hasDefault = false;

  // TODO: Cover the enum case once switchOp comes out with input as enum
  for (auto caseOp : cases) {
    if (caseOp.getKind() == cir::CaseOpKind::Default) {
      hasDefault = true;
      break;
    }
  }

  return hasDefault;
}
static bool
ignoreDefaultsWithCoveredEnums(mlir::Block *block,
                               mlir::DenseSet<mlir::Block *> &shouldNotVisit) {
  mlir::Operation *potentialSwitchOp = block->getParentOp();
  if (shouldNotVisit.contains(block))
    return true;
  if (cir::SwitchOp switchOp =
          llvm::dyn_cast_or_null<cir::SwitchOp>(potentialSwitchOp)) {
    if (switchDefaultsWithCoveredEnums(switchOp)) {
      switchOp->walk(
          [&shouldNotVisit](mlir::Block *op) { shouldNotVisit.insert(op); });
    }
    return true;
  }

  return false;
}
ControlFlowKind FallThroughWarningPass::checkFallThrough(cir::FuncOp cfg) {

  assert(cfg && "there can't be a null func op");

  // TODO: Is no CFG akin to a declaration?
  if (cfg.isDeclaration()) {
    return UnknownFallThrough;
  }

  mlir::SetVector<mlir::Block *> liveSet = this->getLiveSet(cfg);

  bool hasLiveReturn = false;
  bool hasFakeEdge = false;
  bool hasPlainEdge = false;
  bool hasAbnormalEdge = false;

  // This corresponds to OG's IgnoreDefaultsWithCoveredEnums
  mlir::DenseSet<mlir::Block *> shouldNotVisit;

  // INFO: in OG clang CFG, they have an empty exit block, so when they query
  // pred of exit OG, they get all exit blocks
  //
  // I guess in CIR, we can pretend exit blocks are all blocks that have no
  // successor?
  for (mlir::Block *pred : liveSet) {
    if (ignoreDefaultsWithCoveredEnums(pred, shouldNotVisit))
      continue;

    // We consider no successor as 'exit blocks'
    if (!pred->hasNoSuccessors())
      continue;

    // Walk all ReturnOp operations to find returns in nested regions

    if (!pred->mightHaveTerminator())
      continue;

    mlir::Operation *term = pred->getTerminator();

    LLVM_DEBUG(pred->dump());

    // TODO: hasNoReturnElement() in OG here, not sure how to work it in here
    // yet

    // INFO: In OG, we'll be looking for destructor since it can appear past
    // return but i guess not in CIR? In this case we'll only be examining the
    // terminator

    // INFO: OG clang has this equals true whenever ri == re, which means this
    // is true only when a block only has the terminator, or its size is 1.
    //
    // equivalent is std::distance(pred.begin(), pred.end()) == 1;

    if (auto returnOp = dyn_cast<cir::ReturnOp>(term)) {
      if (!isPhonyReturn(returnOp)) {
        hasLiveReturn = true;
        continue;
      }
    }

    if (isa<cir::TryOp>(term)) {
      hasAbnormalEdge = true;
      continue;
    }

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
  // noreturn, that don't return.  If people would like this warning to be
  // more accurate, such functions should be marked as noreturn.
  //
  // llvm_unreachable("");
  return AlwaysFallThrough;
}
} // namespace clang
