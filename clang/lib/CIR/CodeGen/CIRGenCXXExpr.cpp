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

#include "CIRGenCXXABI.h"
#include "CIRGenFunction.h"

#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/CIR/MissingFeatures.h"

using namespace clang;
using namespace clang::CIRGen;

namespace {
struct MemberCallInfo {
  RequiredArgs reqArgs;
  // Number of prefix arguments for the call. Ignores the `this` pointer.
  unsigned prefixSize;
};
} // namespace

static MemberCallInfo commonBuildCXXMemberOrOperatorCall(
    CIRGenFunction &cgf, const CXXMethodDecl *md, mlir::Value thisPtr,
    mlir::Value implicitParam, QualType implicitParamTy, const CallExpr *ce,
    CallArgList &args, CallArgList *rtlArgs) {
  assert(ce == nullptr || isa<CXXMemberCallExpr>(ce) ||
         isa<CXXOperatorCallExpr>(ce));
  assert(md->isInstance() &&
         "Trying to emit a member or operator call expr on a static method!");

  // Push the this ptr.
  const CXXRecordDecl *rd =
      cgf.cgm.getCXXABI().getThisArgumentTypeForMethod(md);
  args.add(RValue::get(thisPtr), cgf.getTypes().deriveThisType(rd, md));

  // If there is an implicit parameter (e.g. VTT), emit it.
  if (implicitParam) {
    args.add(RValue::get(implicitParam), implicitParamTy);
  }

  const auto *fpt = md->getType()->castAs<FunctionProtoType>();
  RequiredArgs required =
      RequiredArgs::getFromProtoWithExtraSlots(fpt, args.size());
  unsigned prefixSize = args.size() - 1;

  // Add the rest of the call args
  if (rtlArgs) {
    // Special case: if the caller emitted the arguments right-to-left already
    // (prior to emitting the *this argument), we're done. This happens for
    // assignment operators.
    args.addFrom(*rtlArgs);
  } else if (ce) {
    // Special case: skip first argument of CXXOperatorCall (it is "this").
    unsigned argsToSkip = isa<CXXOperatorCallExpr>(ce) ? 1 : 0;
    cgf.emitCallArgs(args, fpt, drop_begin(ce->arguments(), argsToSkip),
                     ce->getDirectCallee());
  } else {
    assert(
        fpt->getNumParams() == 0 &&
        "No CallExpr specified for function with non-zero number of arguments");
  }

  //  return {required, prefixSize};
  return {required, prefixSize};
}

RValue CIRGenFunction::emitCXXMemberOrOperatorMemberCallExpr(
    const CallExpr *ce, const CXXMethodDecl *md, ReturnValueSlot returnValue,
    bool hasQualifier, NestedNameSpecifier *qualifier, bool isArrow,
    const Expr *base) {
  assert(isa<CXXMemberCallExpr>(ce) || isa<CXXOperatorCallExpr>(ce));

  if (md->isVirtual()) {
    cgm.errorNYI(ce->getSourceRange(),
                 "emitCXXMemberOrOperatorMemberCallExpr: virtual call");
    return RValue::get(nullptr);
  }

  bool trivialForCodegen =
      md->isTrivial() || (md->isDefaulted() && md->getParent()->isUnion());
  bool trivialAssignment =
      trivialForCodegen &&
      (md->isCopyAssignmentOperator() || md->isMoveAssignmentOperator()) &&
      !md->getParent()->mayInsertExtraPadding();
  (void)trivialAssignment;

  // C++17 demands that we evaluate the RHS of a (possibly-compound) assignment
  // operator before the LHS.
  CallArgList rtlArgStorage;
  CallArgList *rtlArgs = nullptr;
  if (auto *oce = dyn_cast<CXXOperatorCallExpr>(ce)) {
    if (oce->isAssignmentOp()) {
      cgm.errorNYI(
          oce->getSourceRange(),
          "emitCXXMemberOrOperatorMemberCallExpr: assignment operator");
    }
  }

  LValue thisPtr;
  if (isArrow) {
    LValueBaseInfo baseInfo;
    assert(!cir::MissingFeatures::opTBAA());
    Address thisValue = emitPointerWithAlignment(base, &baseInfo);
    thisPtr = makeAddrLValue(thisValue, base->getType(), baseInfo);
  } else {
    thisPtr = emitLValue(base);
  }

  if (isa<CXXConstructorDecl>(md)) {
    cgm.errorNYI(ce->getSourceRange(),
                 "emitCXXMemberOrOperatorMemberCallExpr: constructor call");
    return RValue::get(nullptr);
  }

  if (trivialForCodegen) {
    if (isa<CXXDestructorDecl>(md))
      return RValue::get(nullptr);

    if (trivialAssignment) {
      cgm.errorNYI(ce->getSourceRange(),
                   "emitCXXMemberOrOperatorMemberCallExpr: trivial assignment");
      return RValue::get(nullptr);
    }

    assert(md->getParent()->mayInsertExtraPadding() &&
           "unknown trivial member function");
  }

  // Compute the function type we're calling
  const CXXMethodDecl *calleeDecl = md;
  const CIRGenFunctionInfo *fInfo = nullptr;
  if (isa<CXXDestructorDecl>(calleeDecl)) {
    cgm.errorNYI(ce->getSourceRange(),
                 "emitCXXMemberOrOperatorMemberCallExpr: destructor call");
    return RValue::get(nullptr);
  }

  fInfo = &cgm.getTypes().arrangeCXXMethodDeclaration(calleeDecl);

  mlir::Type ty = cgm.getTypes().getFunctionType(*fInfo);

  assert(!cir::MissingFeatures::sanitizers());
  assert(!cir::MissingFeatures::emitTypeCheck());

  if (isa<CXXDestructorDecl>(calleeDecl)) {
    cgm.errorNYI(ce->getSourceRange(),
                 "emitCXXMemberOrOperatorMemberCallExpr: destructor call");
    return RValue::get(nullptr);
  }

  assert(!cir::MissingFeatures::sanitizers());
  if (getLangOpts().AppleKext) {
    cgm.errorNYI(ce->getSourceRange(),
                 "emitCXXMemberOrOperatorMemberCallExpr: AppleKext");
    return RValue::get(nullptr);
  }
  CIRGenCallee callee =
      CIRGenCallee::forDirect(cgm.getAddrOfFunction(md, ty), GlobalDecl(md));

  return emitCXXMemberOrOperatorCall(
      calleeDecl, callee, returnValue, thisPtr.getPointer(),
      /*ImplicitParam=*/nullptr, QualType(), ce, rtlArgs);
}

RValue
CIRGenFunction::emitCXXOperatorMemberCallExpr(const CXXOperatorCallExpr *e,
                                              const CXXMethodDecl *md,
                                              ReturnValueSlot returnValue) {
  assert(md->isInstance() &&
         "Trying to emit a member call expr on a static method!");
  return emitCXXMemberOrOperatorMemberCallExpr(
      e, md, returnValue, /*HasQualifier=*/false, /*Qualifier=*/nullptr,
      /*IsArrow=*/false, e->getArg(0));
}

RValue CIRGenFunction::emitCXXMemberOrOperatorCall(
    const CXXMethodDecl *md, const CIRGenCallee &callee,
    ReturnValueSlot returnValue, mlir::Value thisPtr, mlir::Value implicitParam,
    QualType implicitParamTy, const CallExpr *ce, CallArgList *rtlArgs) {
  const auto *fpt = md->getType()->castAs<FunctionProtoType>();
  CallArgList args;
  MemberCallInfo callInfo = commonBuildCXXMemberOrOperatorCall(
      *this, md, thisPtr, implicitParam, implicitParamTy, ce, args, rtlArgs);
  auto &fnInfo = cgm.getTypes().arrangeCXXMethodCall(
      args, fpt, callInfo.reqArgs, callInfo.prefixSize);
  assert((ce || currSrcLoc) && "expected source location");
  mlir::Location loc = ce ? getLoc(ce->getExprLoc()) : *currSrcLoc;
  assert(!cir::MissingFeatures::opCallMustTail());
  return emitCall(fnInfo, callee, returnValue, args, nullptr, loc);
}
