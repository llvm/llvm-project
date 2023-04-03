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
    // Special case: if the caller emitted the arguments right-to-left already
    // (prior to emitting the *this argument), we're done. This happens for
    // assignment operators.
    Args.addFrom(*RtlArgs);
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
  assert((CE || currSrcLoc) && "expected source location");
  mlir::Location loc = CE ? getLoc(CE->getExprLoc()) : *currSrcLoc;
  return buildCall(FnInfo, Callee, ReturnValue, Args, nullptr,
                   CE && CE == MustTailCall, loc);
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
    if (OCE->isAssignmentOp()) {
      // See further note on TrivialAssignment, we don't handle this during
      // codegen, differently than LLVM, which early optimizes like this:
      //  if (TrivialAssignment) {
      //    TrivialAssignmentRHS = buildLValue(CE->getArg(1));
      //  } else {
      RtlArgs = &RtlArgStorage;
      buildCallArgs(*RtlArgs, MD->getType()->castAs<FunctionProtoType>(),
                    drop_begin(CE->arguments(), 1), CE->getDirectCallee(),
                    /*ParamsToSkip*/ 0, EvaluationOrder::ForceRightToLeft);
    }
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
    if (isa<CXXDestructorDecl>(MD))
      return RValue::get(nullptr);

    if (TrivialAssignment) {
      // From LLVM codegen:
      // We don't like to generate the trivial copy/move assignment operator
      // when it isn't necessary; just produce the proper effect here.
      // It's important that we use the result of EmitLValue here rather than
      // emitting call arguments, in order to preserve TBAA information from
      // the RHS.
      //
      // We don't early optimize like LLVM does:
      // LValue RHS = isa<CXXOperatorCallExpr>(CE) ? TrivialAssignmentRHS
      //                                           :
      //                                           buildLValue(*CE->arg_begin());
      // buildAggregateAssign(This, RHS, CE->getType());
      // return RValue::get(This.getPointer());
    } else {
      assert(MD->getParent()->mayInsertExtraPadding() &&
             "unknown trivial member function");
    }
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

RValue
CIRGenFunction::buildCXXOperatorMemberCallExpr(const CXXOperatorCallExpr *E,
                                               const CXXMethodDecl *MD,
                                               ReturnValueSlot ReturnValue) {
  assert(MD->isInstance() &&
         "Trying to emit a member call expr on a static method!");
  return buildCXXMemberOrOperatorMemberCallExpr(
      E, MD, ReturnValue, /*HasQualifier=*/false, /*Qualifier=*/nullptr,
      /*IsArrow=*/false, E->getArg(0));
}

void CIRGenFunction::buildCXXConstructExpr(const CXXConstructExpr *E,
                                           AggValueSlot Dest) {
  assert(!Dest.isIgnored() && "Must have a destination!");
  const auto *CD = E->getConstructor();

  // If we require zero initialization before (or instead of) calling the
  // constructor, as can be the case with a non-user-provided default
  // constructor, emit the zero initialization now, unless destination is
  // already zeroed.
  if (E->requiresZeroInitialization() && !Dest.isZeroed()) {
    switch (E->getConstructionKind()) {
    case CXXConstructionKind::Delegating:
    case CXXConstructionKind::Complete:
      buildNullInitialization(getLoc(E->getSourceRange()), Dest.getAddress(),
                              E->getType());
      break;
    case CXXConstructionKind::VirtualBase:
    case CXXConstructionKind::NonVirtualBase:
      llvm_unreachable("NYI");
      break;
    }
  }

  // If this is a call to a trivial default constructor:
  // In LLVM: do nothing.
  // In CIR: emit as a regular call, other later passes should lower the
  // ctor call into trivial initialization.
  // if (CD->isTrivial() && CD->isDefaultConstructor())
  //  return;

  // Elide the constructor if we're constructing from a temporary
  if (getLangOpts().ElideConstructors && E->isElidable()) {
    // FIXME: This only handles the simplest case, where the source object is
    //        passed directly as the first argument to the constructor. This
    //        should also handle stepping through implicit casts and conversion
    //        sequences which involve two steps, with a conversion operator
    //        follwed by a converting constructor.
    const auto *SrcObj = E->getArg(0);
    assert(SrcObj->isTemporaryObject(getContext(), CD->getParent()));
    assert(
        getContext().hasSameUnqualifiedType(E->getType(), SrcObj->getType()));
    buildAggExpr(SrcObj, Dest);
    return;
  }

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
    llvm_unreachable("NYI");
  case CXXConstructionKind::VirtualBase:
    llvm_unreachable("NYI");
  case CXXConstructionKind::NonVirtualBase:
    Type = Ctor_Base;
    break;
  }

  buildCXXConstructorCall(CD, Type, ForVirtualBase, Delegating, Dest, E);
}

mlir::Value CIRGenFunction::buildCXXNewExpr(const CXXNewExpr *E) {
  assert(0 && "not implemented");
}
