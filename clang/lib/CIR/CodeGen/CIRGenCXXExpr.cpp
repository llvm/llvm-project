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
    bool hasQualifier, NestedNameSpecifier qualifier, bool isArrow,
    const Expr *base) {
  assert(isa<CXXMemberCallExpr>(ce) || isa<CXXOperatorCallExpr>(ce));

  if (md->isVirtual()) {
    cgm.errorNYI(ce->getSourceRange(),
                 "emitCXXMemberOrOperatorMemberCallExpr: virtual call");
    return RValue::get(nullptr);
  }

  // Note on trivial assignment
  // --------------------------
  // Classic codegen avoids generating the trivial copy/move assignment operator
  // when it isn't necessary, choosing instead to just produce IR with an
  // equivalent effect. We have chosen not to do that in CIR, instead emitting
  // trivial copy/move assignment operators and allowing later transformations
  // to optimize them away if appropriate.

  // C++17 demands that we evaluate the RHS of a (possibly-compound) assignment
  // operator before the LHS.
  CallArgList rtlArgStorage;
  CallArgList *rtlArgs = nullptr;
  if (auto *oce = dyn_cast<CXXOperatorCallExpr>(ce)) {
    if (oce->isAssignmentOp()) {
      rtlArgs = &rtlArgStorage;
      emitCallArgs(*rtlArgs, md->getType()->castAs<FunctionProtoType>(),
                   drop_begin(ce->arguments(), 1), ce->getDirectCallee(),
                   /*ParamsToSkip*/ 0);
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

  if ((md->isTrivial() || (md->isDefaulted() && md->getParent()->isUnion())) &&
      isa<CXXDestructorDecl>(md))
    return RValue::get(nullptr);

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
      e, md, returnValue, /*HasQualifier=*/false, /*Qualifier=*/std::nullopt,
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

static mlir::Value emitCXXNewAllocSize(CIRGenFunction &cgf, const CXXNewExpr *e,
                                       unsigned minElements,
                                       mlir::Value &numElements,
                                       mlir::Value &sizeWithoutCookie) {
  QualType type = e->getAllocatedType();
  mlir::Location loc = cgf.getLoc(e->getSourceRange());

  if (!e->isArray()) {
    CharUnits typeSize = cgf.getContext().getTypeSizeInChars(type);
    sizeWithoutCookie = cgf.getBuilder().getConstant(
        loc, cir::IntAttr::get(cgf.SizeTy, typeSize.getQuantity()));
    return sizeWithoutCookie;
  }

  cgf.cgm.errorNYI(e->getSourceRange(), "emitCXXNewAllocSize: array");
  return {};
}

static void storeAnyExprIntoOneUnit(CIRGenFunction &cgf, const Expr *init,
                                    QualType allocType, Address newPtr,
                                    AggValueSlot::Overlap_t mayOverlap) {
  // FIXME: Refactor with emitExprAsInit.
  switch (cgf.getEvaluationKind(allocType)) {
  case cir::TEK_Scalar:
    cgf.emitScalarInit(init, cgf.getLoc(init->getSourceRange()),
                       cgf.makeAddrLValue(newPtr, allocType), false);
    return;
  case cir::TEK_Complex:
    cgf.cgm.errorNYI(init->getSourceRange(),
                     "storeAnyExprIntoOneUnit: complex");
    return;
  case cir::TEK_Aggregate: {
    assert(!cir::MissingFeatures::aggValueSlotGC());
    assert(!cir::MissingFeatures::sanitizers());
    AggValueSlot slot = AggValueSlot::forAddr(
        newPtr, allocType.getQualifiers(), AggValueSlot::IsDestructed,
        AggValueSlot::IsNotAliased, mayOverlap, AggValueSlot::IsNotZeroed);
    cgf.emitAggExpr(init, slot);
    return;
  }
  }
  llvm_unreachable("bad evaluation kind");
}

static void emitNewInitializer(CIRGenFunction &cgf, const CXXNewExpr *e,
                               QualType elementType, mlir::Type elementTy,
                               Address newPtr, mlir::Value numElements,
                               mlir::Value allocSizeWithoutCookie) {
  assert(!cir::MissingFeatures::generateDebugInfo());
  if (e->isArray()) {
    cgf.cgm.errorNYI(e->getSourceRange(), "emitNewInitializer: array");
  } else if (const Expr *init = e->getInitializer()) {
    storeAnyExprIntoOneUnit(cgf, init, e->getAllocatedType(), newPtr,
                            AggValueSlot::DoesNotOverlap);
  }
}

RValue CIRGenFunction::emitCXXDestructorCall(
    GlobalDecl dtor, const CIRGenCallee &callee, mlir::Value thisVal,
    QualType thisTy, mlir::Value implicitParam, QualType implicitParamTy,
    const CallExpr *ce) {
  const CXXMethodDecl *dtorDecl = cast<CXXMethodDecl>(dtor.getDecl());

  assert(!thisTy.isNull());
  assert(thisTy->getAsCXXRecordDecl() == dtorDecl->getParent() &&
         "Pointer/Object mixup");

  assert(!cir::MissingFeatures::addressSpace());

  CallArgList args;
  commonBuildCXXMemberOrOperatorCall(*this, dtorDecl, thisVal, implicitParam,
                                     implicitParamTy, ce, args, nullptr);
  assert((ce || dtor.getDecl()) && "expected source location provider");
  assert(!cir::MissingFeatures::opCallMustTail());
  return emitCall(cgm.getTypes().arrangeCXXStructorDeclaration(dtor), callee,
                  ReturnValueSlot(), args, nullptr,
                  ce ? getLoc(ce->getExprLoc())
                     : getLoc(dtor.getDecl()->getSourceRange()));
}

/// Emit a call to an operator new or operator delete function, as implicitly
/// created by new-expressions and delete-expressions.
static RValue emitNewDeleteCall(CIRGenFunction &cgf,
                                const FunctionDecl *calleeDecl,
                                const FunctionProtoType *calleeType,
                                const CallArgList &args) {
  cir::CIRCallOpInterface callOrTryCall;
  cir::FuncOp calleePtr = cgf.cgm.getAddrOfFunction(calleeDecl);
  CIRGenCallee callee =
      CIRGenCallee::forDirect(calleePtr, GlobalDecl(calleeDecl));
  RValue rv =
      cgf.emitCall(cgf.cgm.getTypes().arrangeFreeFunctionCall(args, calleeType),
                   callee, ReturnValueSlot(), args, &callOrTryCall);

  /// C++1y [expr.new]p10:
  ///   [In a new-expression,] an implementation is allowed to omit a call
  ///   to a replaceable global allocation function.
  ///
  /// We model such elidable calls with the 'builtin' attribute.
  assert(!cir::MissingFeatures::attributeBuiltin());
  return rv;
}

mlir::Value CIRGenFunction::emitCXXNewExpr(const CXXNewExpr *e) {
  // The element type being allocated.
  QualType allocType = getContext().getBaseElementType(e->getAllocatedType());

  // 1. Build a call to the allocation function.
  FunctionDecl *allocator = e->getOperatorNew();

  // If there is a brace-initializer, cannot allocate fewer elements than inits.
  unsigned minElements = 0;
  if (e->isArray() && e->hasInitializer()) {
    cgm.errorNYI(e->getSourceRange(), "emitCXXNewExpr: array initializer");
  }

  mlir::Value numElements = nullptr;
  mlir::Value allocSizeWithoutCookie = nullptr;
  mlir::Value allocSize = emitCXXNewAllocSize(
      *this, e, minElements, numElements, allocSizeWithoutCookie);
  CharUnits allocAlign = getContext().getTypeAlignInChars(allocType);

  // Emit the allocation call.
  Address allocation = Address::invalid();
  CallArgList allocatorArgs;
  if (allocator->isReservedGlobalPlacementOperator()) {
    cgm.errorNYI(e->getSourceRange(),
                 "emitCXXNewExpr: reserved global placement operator");
  } else {
    const FunctionProtoType *allocatorType =
        allocator->getType()->castAs<FunctionProtoType>();
    unsigned paramsToSkip = 0;

    // The allocation size is the first argument.
    QualType sizeType = getContext().getSizeType();
    allocatorArgs.add(RValue::get(allocSize), sizeType);
    ++paramsToSkip;

    if (allocSize != allocSizeWithoutCookie) {
      CharUnits cookieAlign = getSizeAlign(); // FIXME: Ask the ABI.
      allocAlign = std::max(allocAlign, cookieAlign);
    }

    // The allocation alignment may be passed as the second argument.
    if (e->passAlignment()) {
      cgm.errorNYI(e->getSourceRange(), "emitCXXNewExpr: pass alignment");
    }

    // FIXME: Why do we not pass a CalleeDecl here?
    emitCallArgs(allocatorArgs, allocatorType, e->placement_arguments(),
                 AbstractCallee(), paramsToSkip);
    RValue rv =
        emitNewDeleteCall(*this, allocator, allocatorType, allocatorArgs);

    // Set !heapallocsite metadata on the call to operator new.
    assert(!cir::MissingFeatures::generateDebugInfo());

    // If this was a call to a global replaceable allocation function that does
    // not take an alignment argument, the allocator is known to produce storage
    // that's suitably aligned for any object that fits, up to a known
    // threshold. Otherwise assume it's suitably aligned for the allocated type.
    CharUnits allocationAlign = allocAlign;
    if (!e->passAlignment() &&
        allocator->isReplaceableGlobalAllocationFunction()) {
      const TargetInfo &target = cgm.getASTContext().getTargetInfo();
      unsigned allocatorAlign = llvm::bit_floor(std::min<uint64_t>(
          target.getNewAlign(), getContext().getTypeSize(allocType)));
      allocationAlign = std::max(
          allocationAlign, getContext().toCharUnitsFromBits(allocatorAlign));
    }

    mlir::Value allocPtr = rv.getValue();
    allocation = Address(
        allocPtr, mlir::cast<cir::PointerType>(allocPtr.getType()).getPointee(),
        allocationAlign);
  }

  // Emit a null check on the allocation result if the allocation
  // function is allowed to return null (because it has a non-throwing
  // exception spec or is the reserved placement new) and we have an
  // interesting initializer will be running sanitizers on the initialization.
  bool nullCheck = e->shouldNullCheckAllocation() &&
                   (!allocType.isPODType(getContext()) || e->hasInitializer());
  assert(!cir::MissingFeatures::exprNewNullCheck());
  if (nullCheck)
    cgm.errorNYI(e->getSourceRange(), "emitCXXNewExpr: null check");

  // If there's an operator delete, enter a cleanup to call it if an
  // exception is thrown.
  if (e->getOperatorDelete() &&
      !e->getOperatorDelete()->isReservedGlobalPlacementOperator())
    cgm.errorNYI(e->getSourceRange(), "emitCXXNewExpr: operator delete");

  if (allocSize != allocSizeWithoutCookie)
    cgm.errorNYI(e->getSourceRange(), "emitCXXNewExpr: array with cookies");

  mlir::Type elementTy = convertTypeForMem(allocType);
  Address result = builder.createElementBitCast(getLoc(e->getSourceRange()),
                                                allocation, elementTy);

  // Passing pointer through launder.invariant.group to avoid propagation of
  // vptrs information which may be included in previous type.
  // To not break LTO with different optimizations levels, we do it regardless
  // of optimization level.
  if (cgm.getCodeGenOpts().StrictVTablePointers &&
      allocator->isReservedGlobalPlacementOperator())
    cgm.errorNYI(e->getSourceRange(), "emitCXXNewExpr: strict vtable pointers");

  assert(!cir::MissingFeatures::sanitizers());

  emitNewInitializer(*this, e, allocType, elementTy, result, numElements,
                     allocSizeWithoutCookie);
  return result.getPointer();
}
