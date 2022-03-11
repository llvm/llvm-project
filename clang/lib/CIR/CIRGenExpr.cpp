#include "CIRGenCall.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"

#include "clang/AST/GlobalDecl.h"

#include "mlir/IR/Value.h"

using namespace cir;
using namespace clang;

static mlir::FuncOp buildFunctionDeclPointer(CIRGenModule &CGM, GlobalDecl GD) {
  const auto *FD = cast<FunctionDecl>(GD.getDecl());
  assert(!FD->hasAttr<WeakRefAttr>() && "NYI");

  auto V = CGM.GetAddrOfFunction(GD);
  assert(FD->hasPrototype() &&
         "Only prototyped functions are currently callable");

  return V;
}

static CIRGenCallee buildDirectCallee(CIRGenFunction &CGF, GlobalDecl GD) {
  const auto *FD = cast<FunctionDecl>(GD.getDecl());

  assert(!FD->getBuiltinID() && "Builtins NYI");

  auto CalleePtr = buildFunctionDeclPointer(CGF.CGM, GD);

  assert(!CGF.CGM.getLangOpts().CUDA && "NYI");

  return CIRGenCallee::forDirect(CalleePtr, GD);
}

CIRGenCallee CIRGenFunction::buildCallee(const clang::Expr *E) {
  E = E->IgnoreParens();

  if (auto ICE = dyn_cast<ImplicitCastExpr>(E)) {
    assert(ICE && "Only ICE supported so far!");
    assert(ICE->getCastKind() == CK_FunctionToPointerDecay &&
           "No other casts supported yet");

    return buildCallee(ICE->getSubExpr());
  } else if (auto DRE = dyn_cast<DeclRefExpr>(E)) {
    auto FD = dyn_cast<FunctionDecl>(DRE->getDecl());
    assert(FD &&
           "DeclRef referring to FunctionDecl onlything supported so far");
    return buildDirectCallee(*this, FD);
  }

  assert(!dyn_cast<MemberExpr>(E) && "NYI");
  assert(!dyn_cast<SubstNonTypeTemplateParmExpr>(E) && "NYI");
  assert(!dyn_cast<CXXPseudoDestructorExpr>(E) && "NYI");

  assert(false && "Nothing else supported yet!");
}
