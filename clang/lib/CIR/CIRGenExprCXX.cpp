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
