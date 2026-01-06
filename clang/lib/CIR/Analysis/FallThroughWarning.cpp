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
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <unordered_map>
#include <unordered_set>

#define DEBUG_TYPE "cir-fallthrough"

using namespace mlir;
using namespace cir;
using namespace clang;

namespace clang {

class FallthroughInstance {
  std::unordered_map<mlir::Operation *, ControlFlowKind> fallthroughOp;

  ControlFlowKind getCfkFor(mlir::Operation *op) {
    auto it = fallthroughOp.find(op);
    if (it != fallthroughOp.end())
      return it->second;
    return ControlFlowKind::Undetermined;
  }
  ControlFlowKind updateControlFlowKind(mlir::Operation *parentOp,
                                        ControlFlowKind cfk,
                                        ControlFlowKind newCfk) {
    cfk = std::min(cfk, newCfk);

    if (parentOp) {
      auto x = fallthroughOp.find(parentOp);
      if (x != fallthroughOp.end())
        fallthroughOp[parentOp] = std::min(cfk, x->second);
      else
        fallthroughOp[parentOp] = cfk;
    }
    return cfk;
  }

  ControlFlowKind maybeFallThrough() {
    return ControlFlowKind::MaybeFallThrough;
  }

public:
  ControlFlowKind isFallThroughAble(mlir::Operation &op);
  ControlFlowKind isFallThroughAble(mlir::Block &block);
  ControlFlowKind handleFallthroughCaseOp(cir::CaseOp caseOp);
  ControlFlowKind handleFallthroughScopeOp(cir::ScopeOp scopeOp);
  ControlFlowKind handleFallthroughSwitchOp(cir::SwitchOp swOp);
  ControlFlowKind handleFallthroughFuncOp(cir::FuncOp fnOp);
};

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
ControlFlowKind
FallthroughInstance::handleFallthroughCaseOp(cir::CaseOp caseOp) {
  ControlFlowKind cfk = NeverFallThroughOrReturn;
  for (auto &block : caseOp.getCaseRegion()) {
    cfk = updateControlFlowKind(caseOp, cfk, isFallThroughAble(block));
  }

  return cfk;
}

ControlFlowKind
FallthroughInstance::handleFallthroughSwitchOp(cir::SwitchOp swOp) {
  // go through each cases
  llvm::SmallVector<CaseOp> cases;
  swOp.collectCases(cases);

  bool coverAllCases = swOp.getAllEnumCasesCovered();
  bool coverDefault = false;

  ControlFlowKind cfk = Undetermined;

  // Default case detection
  for (auto caseOp : cases) {
    if (caseOp.getKind() == CaseOpKind::Default) {
      coverDefault = true;
      break;
    }
  }
  // if we don't cover all case and we don't have a default -> fall through
  if (!coverAllCases && !coverDefault) {
    cfk = maybeFallThrough();
  } else {
    for (auto caseOp : cases) {
      ControlFlowKind fallthrough = handleFallthroughCaseOp(caseOp);
      cfk = updateControlFlowKind(swOp, cfk, fallthrough);
    }
  }

  return cfk;
}
ControlFlowKind
FallthroughInstance::handleFallthroughScopeOp(cir::ScopeOp scopeOp) {
  ControlFlowKind cfk = Undetermined;
  for (auto &block : scopeOp.getScopeRegion()) {
    cfk = updateControlFlowKind(scopeOp, cfk, isFallThroughAble(block));
  }
  return cfk;
}

ControlFlowKind FallthroughInstance::isFallThroughAble(mlir::Operation &op) {
  if (auto swOp = mlir::dyn_cast_or_null<cir::SwitchOp>(op)) {
    return handleFallthroughSwitchOp(swOp);
  }
  if (auto caseOp = mlir::dyn_cast_or_null<cir::CaseOp>(op)) {
    return handleFallthroughCaseOp(caseOp);
  }
  if (auto scopeOp = mlir::dyn_cast_or_null<cir::ScopeOp>(op)) {
    return handleFallthroughScopeOp(scopeOp);
  }

  auto *parentOp = op.getParentOp();
  ControlFlowKind cfk = getCfkFor(parentOp);
  if (auto returnOp = mlir::dyn_cast_or_null<cir::ReturnOp>(op)) {
    if (isPhonyReturn(returnOp)) {
      if (cfk < ControlFlowKind::NeverFallThrough || cfk == Undetermined) {
        return maybeFallThrough();
      }
    }
    return ControlFlowKind::NeverFallThrough;
  }
  if (auto yieldOp = mlir::dyn_cast_or_null<cir::YieldOp>(op)) {
    LLVM_DEBUG({
      llvm::errs() << "Encountered yield op\n";
      yieldOp->dump();
      yieldOp.getLoc().dump();
    });
    if (isa<cir::CaseOp, cir::ScopeOp>(parentOp))
      if (cfk < ControlFlowKind::NeverFallThrough|| cfk == Undetermined) 
          return maybeFallThrough();
    return ControlFlowKind::Undetermined;
  }
  if (auto breakOp = mlir::dyn_cast_or_null<cir::BreakOp>(op)) {
    auto *parentOp = breakOp->getParentOp();
    if (isa<cir::CaseOp>(parentOp)) {
      return maybeFallThrough();
    }
  }
  return ControlFlowKind::Undetermined;
}

ControlFlowKind FallthroughInstance::isFallThroughAble(mlir::Block &block) {
  ControlFlowKind cfk = Undetermined;
  for (auto &op : block) {
    cfk =
        updateControlFlowKind(block.getParentOp(), cfk, isFallThroughAble(op));

    if (cfk < ControlFlowKind::NeverFallThrough) {
      return cfk;
    }
  }
  return cfk;
}

// TODO: Add a class for fall through config later
ControlFlowKind FallthroughInstance::handleFallthroughFuncOp(cir::FuncOp fnOp) {
  ControlFlowKind cfk = Undetermined;
  for (auto &o : fnOp) {
    cfk = updateControlFlowKind(fnOp, cfk, isFallThroughAble(o));
  }
  return cfk;
}

void FallThroughWarningPass::checkFallThroughForFuncBody(
    ASTContext& astContext, DiagnosticsEngine &diags, cir::FuncOp cfg, QualType blockType,
    const CheckFallThroughDiagnostics &cd) {

  auto *d = getDeclByName(astContext, cfg.getName());
  assert(d && "we need non null decl");
  auto *body = d->getBody();

  // Functions without bodies (declarations only) don't need fall-through
  // analysis
  if (!body)
    return;

  bool returnsVoid = false;
  bool hasNoReturn = false;

  SourceLocation rBrace = body->getEndLoc();
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

  // Short circuit for compilation speed.
  if (cd.checkDiagnostics(diags, returnsVoid, hasNoReturn))
    return;

  // cpu_dispatch functions permit empty function bodies for ICC compatibility.
  // TODO: Do we have isCPUDispatchMultiVersion?
  FallthroughInstance fi;
  ControlFlowKind fallThroughType = fi.handleFallthroughFuncOp(cfg);

  if (fallThroughType < ControlFlowKind::NeverFallThrough) {
    LLVM_DEBUG({ llvm::errs() << "Fall through detected\n"; });
  }

  switch (fallThroughType) {
  case Undetermined:
    [[fallthrough]];
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

      diags.Report(rBrace, cd.diagFallThroughReturnsNonVoid)
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

} // namespace clang
