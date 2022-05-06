//===--- CIRGenExprCXX.cpp - Emit CIR Code for C++ expressions ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with code generation of C++ expressions
//
//===----------------------------------------------------------------------===//

#include <CIRGenCXXABI.h>
#include <CIRGenFunction.h>
#include <CIRGenModule.h>
#include <CIRGenValue.h>
#include <UnimplementedFeatureGuarding.h>

#include <clang/AST/DeclCXX.h>

using namespace cir;
using namespace clang;

namespace {
struct MemberCallInfo {
  RequiredArgs ReqArgs;
  // Number of prefix arguments for the call. Ignores the `this` pointer.
  unsigned PrefixSize;
};
} // namespace

static MemberCallInfo
commonBuildCXXMemberOrOperatorCall(CIRGenFunction &CGF, const CXXMethodDecl *MD,
                                   mlir::Value This, mlir::Value ImplicitParam,
                                   QualType ImplicitParamTy, const CallExpr *CE,
                                   CallArgList &Args, CallArgList *RtlArgs) {
  assert(CE == nullptr || isa<CXXMemberCallExpr>(CE) ||
         isa<CXXOperatorCallExpr>(CE));
  assert(MD->isInstance() &&
         "Trying to emit a member or operator call expr on a static method!");

  // Push the this ptr.
  const CXXRecordDecl *RD =
      CGF.CGM.getCXXABI().getThisArgumentTypeForMethod(MD);
  Args.add(RValue::get(This), CGF.getTypes().DeriveThisType(RD, MD));

  // If there is an implicit parameter (e.g. VTT), emit it.
  if (ImplicitParam) {
    llvm_unreachable("NYI");
  }

  const auto *FPT = MD->getType()->castAs<FunctionProtoType>();
  RequiredArgs required = RequiredArgs::forPrototypePlus(FPT, Args.size());
  unsigned PrefixSize = Args.size() - 1;

  // Add the rest of the call args
  if (RtlArgs) {
    llvm_unreachable("NYI");
  } else if (CE) {
    // Special case: skip first argument of CXXOperatorCall (it is "this").
    unsigned ArgsToSkip = isa<CXXOperatorCallExpr>(CE) ? 1 : 0;
    CGF.buildCallArgs(Args, FPT, drop_begin(CE->arguments(), ArgsToSkip),
                      CE->getDirectCallee());
  } else {
    assert(
        FPT->getNumParams() == 0 &&
        "No CallExpr specified for function with non-zero number of arguments");
  }

  return {required, PrefixSize};
}

RValue CIRGenFunction::buildCXXMemberOrOperatorCall(
    const CXXMethodDecl *MD, const CIRGenCallee &Callee,
    ReturnValueSlot ReturnValue, mlir::Value This, mlir::Value ImplicitParam,
    QualType ImplicitParamTy, const CallExpr *CE, CallArgList *RtlArgs) {

  const auto *FPT = MD->getType()->castAs<FunctionProtoType>();
  CallArgList Args;
  MemberCallInfo CallInfo = commonBuildCXXMemberOrOperatorCall(
      *this, MD, This, ImplicitParam, ImplicitParamTy, CE, Args, RtlArgs);
  auto &FnInfo = CGM.getTypes().arrangeCXXMethodCall(
      Args, FPT, CallInfo.ReqArgs, CallInfo.PrefixSize);
  return buildCall(FnInfo, Callee, ReturnValue, Args, nullptr,
                   CE && CE == MustTailCall,
                   CE ? CE->getExprLoc() : SourceLocation());
}

RValue CIRGenFunction::buildCXXMemberOrOperatorMemberCallExpr(
    const CallExpr *CE, const CXXMethodDecl *MD, ReturnValueSlot ReturnValue,
    bool HasQualifier, NestedNameSpecifier *Qualifier, bool IsArrow,
    const Expr *Base) {
  assert(isa<CXXMemberCallExpr>(CE) || isa<CXXOperatorCallExpr>(CE));

  // Compute the object pointer.
  bool CanUseVirtualCall = MD->isVirtual() && !HasQualifier;
  assert(!CanUseVirtualCall && "NYI");

  const CXXMethodDecl *DevirtualizedMethod = nullptr;
  if (CanUseVirtualCall &&
      MD->getDevirtualizedMethod(Base, getLangOpts().AppleKext)) {
    llvm_unreachable("NYI");
  }

  bool TrivialForCodegen =
      MD->isTrivial() || (MD->isDefaulted() && MD->getParent()->isUnion());
  bool TrivialAssignment =
      TrivialForCodegen &&
      (MD->isCopyAssignmentOperator() || MD->isMoveAssignmentOperator()) &&
      !MD->getParent()->mayInsertExtraPadding();
  (void)TrivialAssignment;

  // C++17 demands that we evaluate the RHS of a (possibly-compound) assignment
  // operator before the LHS.
  CallArgList RtlArgStorage;
  CallArgList *RtlArgs = nullptr;
  LValue TrivialAssignmentRHS;
  if (auto *OCE = dyn_cast<CXXOperatorCallExpr>(CE)) {
    llvm_unreachable("NYI");
  }

  LValue This;
  if (IsArrow) {
    llvm_unreachable("NYI");
  } else {
    This = buildLValue(Base);
  }

  if (const CXXConstructorDecl *Ctor = dyn_cast<CXXConstructorDecl>(MD)) {
    llvm_unreachable("NYI");
  }

  if (TrivialForCodegen) {
    llvm_unreachable("NYI");
  }

  // Compute the function type we're calling
  const CXXMethodDecl *CalleeDecl =
      DevirtualizedMethod ? DevirtualizedMethod : MD;
  const CIRGenFunctionInfo *FInfo = nullptr;
  if (const auto *Dtor = dyn_cast<CXXDestructorDecl>(CalleeDecl))
    llvm_unreachable("NYI");
  else
    FInfo = &CGM.getTypes().arrangeCXXMethodDeclaration(CalleeDecl);

  mlir::FunctionType Ty = CGM.getTypes().GetFunctionType(*FInfo);

  // C++11 [class.mfct.non-static]p2:
  //   If a non-static member function of a class X is called for an object that
  //   is not of type X, or of a type derived from X, the behavior is undefined.
  SourceLocation CallLoc;
  ASTContext &C = getContext();
  (void)C;
  if (CE)
    CallLoc = CE->getExprLoc();

  SanitizerSet SkippedChecks;
  if (const auto *cmce = dyn_cast<CXXMemberCallExpr>(CE)) {
    auto *ioa = cmce->getImplicitObjectArgument();
    auto isImplicitObjectCXXThis = isWrappedCXXThis(ioa);
    if (isImplicitObjectCXXThis)
      SkippedChecks.set(SanitizerKind::Alignment, true);
    if (isImplicitObjectCXXThis || isa<DeclRefExpr>(ioa))
      SkippedChecks.set(SanitizerKind::Null, true);
  }

  if (UnimplementedFeature::buildTypeCheck())
    llvm_unreachable("NYI");

  // C++ [class.virtual]p12:
  //   Explicit qualification with the scope operator (5.1) suppresses the
  //   virtual call mechanism.
  //
  // We also don't emit a virtual call if the base expression has a record type
  // because then we know what the type is.
  bool useVirtualCall = CanUseVirtualCall && !DevirtualizedMethod;

  if (const auto *dtor = dyn_cast<CXXDestructorDecl>(CalleeDecl)) {
    llvm_unreachable("NYI");
  }

  // FIXME: Uses of 'MD' past this point need to be audited. We may need to use
  // 'CalleeDecl' instead.

  CIRGenCallee Callee;
  if (useVirtualCall) {
    llvm_unreachable("NYI");
  } else {
    if (SanOpts.has(SanitizerKind::CFINVCall)) {
      llvm_unreachable("NYI");
    }

    if (getLangOpts().AppleKext)
      llvm_unreachable("NYI");
    else if (!DevirtualizedMethod)
      Callee = CIRGenCallee::forDirect(CGM.GetAddrOfFunction(MD, Ty),
                                       GlobalDecl(MD));
    else {
      llvm_unreachable("NYI");
    }
  }

  if (MD->isVirtual()) {
    llvm_unreachable("NYI");
  }

  return buildCXXMemberOrOperatorCall(
      CalleeDecl, Callee, ReturnValue, This.getPointer(),
      /*ImplicitParam=*/nullptr, QualType(), CE, RtlArgs);
}

void CIRGenFunction::buildCXXConstructExpr(const CXXConstructExpr *E,
                                           AggValueSlot Dest) {
  assert(!Dest.isIgnored() && "Must have a destination!");
  const auto *CD = E->getConstructor();

  assert(!E->requiresZeroInitialization() && "zero initialization NYI");

  // If this is a call to a trivial default constructor, do nothing.
  if (CD->isTrivial() && CD->isDefaultConstructor())
    assert(!CD->isTrivial() && "trivial constructors NYI");

  assert(!E->isElidable() && "elidable constructors NYI");

  assert(!CGM.getASTContext().getAsArrayType(E->getType()) &&
         "array types NYI");

  clang::CXXCtorType Type = Ctor_Complete;
  bool ForVirtualBase = false;
  bool Delegating = false;

  switch (E->getConstructionKind()) {
  case CXXConstructionKind::Complete:
    Type = Ctor_Complete;
    break;
  case CXXConstructionKind::Delegating:
  case CXXConstructionKind::VirtualBase:
  case CXXConstructionKind::NonVirtualBase:
    assert(false && "Delegating, Virtualbae and NonVirtualBase ctorkind NYI");
  }

  buildCXXConstructorCall(CD, Type, ForVirtualBase, Delegating, Dest, E);
}
