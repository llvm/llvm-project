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
#include "CIRGenConstantEmitter.h"
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

  // Compute the object pointer.
  bool canUseVirtualCall = md->isVirtual() && !hasQualifier;
  const CXXMethodDecl *devirtualizedMethod = nullptr;
  assert(!cir::MissingFeatures::devirtualizeMemberFunction());

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
  const CXXMethodDecl *calleeDecl =
      devirtualizedMethod ? devirtualizedMethod : md;
  const CIRGenFunctionInfo *fInfo = nullptr;
  if (isa<CXXDestructorDecl>(calleeDecl)) {
    cgm.errorNYI(ce->getSourceRange(),
                 "emitCXXMemberOrOperatorMemberCallExpr: destructor call");
    return RValue::get(nullptr);
  }

  fInfo = &cgm.getTypes().arrangeCXXMethodDeclaration(calleeDecl);

  cir::FuncType ty = cgm.getTypes().getFunctionType(*fInfo);

  assert(!cir::MissingFeatures::sanitizers());
  assert(!cir::MissingFeatures::emitTypeCheck());

  // C++ [class.virtual]p12:
  //   Explicit qualification with the scope operator (5.1) suppresses the
  //   virtual call mechanism.
  //
  // We also don't emit a virtual call if the base expression has a record type
  // because then we know what the type is.
  bool useVirtualCall = canUseVirtualCall && !devirtualizedMethod;

  if (isa<CXXDestructorDecl>(calleeDecl)) {
    cgm.errorNYI(ce->getSourceRange(),
                 "emitCXXMemberOrOperatorMemberCallExpr: destructor call");
    return RValue::get(nullptr);
  }

  CIRGenCallee callee;
  if (useVirtualCall) {
    callee = CIRGenCallee::forVirtual(ce, md, thisPtr.getAddress(), ty);
  } else {
    assert(!cir::MissingFeatures::sanitizers());
    if (getLangOpts().AppleKext) {
      cgm.errorNYI(ce->getSourceRange(),
                   "emitCXXMemberOrOperatorMemberCallExpr: AppleKext");
      return RValue::get(nullptr);
    }

    callee = CIRGenCallee::forDirect(cgm.getAddrOfFunction(calleeDecl, ty),
                                     GlobalDecl(calleeDecl));
  }

  if (md->isVirtual()) {
    Address newThisAddr =
        cgm.getCXXABI().adjustThisArgumentForVirtualFunctionCall(
            *this, calleeDecl, thisPtr.getAddress(), useVirtualCall);
    thisPtr.setAddress(newThisAddr);
  }

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

static CharUnits calculateCookiePadding(CIRGenFunction &cgf,
                                        const CXXNewExpr *e) {
  if (!e->isArray())
    return CharUnits::Zero();

  // No cookie is required if the operator new[] being used is the
  // reserved placement operator new[].
  if (e->getOperatorNew()->isReservedGlobalPlacementOperator())
    return CharUnits::Zero();

  return cgf.cgm.getCXXABI().getArrayCookieSize(e);
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

  // The width of size_t.
  unsigned sizeWidth = cgf.cgm.getDataLayout().getTypeSizeInBits(cgf.SizeTy);

  // The number of elements can be have an arbitrary integer type;
  // essentially, we need to multiply it by a constant factor, add a
  // cookie size, and verify that the result is representable as a
  // size_t.  That's just a gloss, though, and it's wrong in one
  // important way: if the count is negative, it's an error even if
  // the cookie size would bring the total size >= 0.
  //
  // If the array size is constant, Sema will have prevented negative
  // values and size overflow.

  // Compute the constant factor.
  llvm::APInt arraySizeMultiplier(sizeWidth, 1);
  while (const ConstantArrayType *cat =
             cgf.getContext().getAsConstantArrayType(type)) {
    type = cat->getElementType();
    arraySizeMultiplier *= cat->getSize();
  }

  CharUnits typeSize = cgf.getContext().getTypeSizeInChars(type);
  llvm::APInt typeSizeMultiplier(sizeWidth, typeSize.getQuantity());
  typeSizeMultiplier *= arraySizeMultiplier;

  // Figure out the cookie size.
  llvm::APInt cookieSize(sizeWidth,
                         calculateCookiePadding(cgf, e).getQuantity());

  // This will be a size_t.
  mlir::Value size;

  // Emit the array size expression.
  // We multiply the size of all dimensions for NumElements.
  // e.g for 'int[2][3]', ElemType is 'int' and NumElements is 6.
  const Expr *arraySize = *e->getArraySize();
  mlir::Attribute constNumElements =
      ConstantEmitter(cgf.cgm, &cgf)
          .emitAbstract(arraySize, arraySize->getType());
  if (constNumElements) {
    // Get an APInt from the constant
    const llvm::APInt &count =
        mlir::cast<cir::IntAttr>(constNumElements).getValue();

    unsigned numElementsWidth = count.getBitWidth();

    // The equivalent code in CodeGen/CGExprCXX.cpp handles these cases as
    // overflow, but that should never happen. The size argument is implicitly
    // cast to a size_t, so it can never be negative and numElementsWidth will
    // always equal sizeWidth.
    assert(!count.isNegative() && "Expected non-negative array size");
    assert(numElementsWidth == sizeWidth &&
           "Expected a size_t array size constant");

    // Okay, compute a count at the right width.
    llvm::APInt adjustedCount = count.zextOrTrunc(sizeWidth);

    // Scale numElements by that.  This might overflow, but we don't
    // care because it only overflows if allocationSize does too, and
    // if that overflows then we shouldn't use this.
    // This emits a constant that may not be used, but we can't tell here
    // whether it will be needed or not.
    numElements =
        cgf.getBuilder().getConstInt(loc, adjustedCount * arraySizeMultiplier);

    // Compute the size before cookie, and track whether it overflowed.
    bool overflow;
    llvm::APInt allocationSize =
        adjustedCount.umul_ov(typeSizeMultiplier, overflow);

    // Sema prevents us from hitting this case
    assert(!overflow && "Overflow in array allocation size");

    // Add in the cookie, and check whether it's overflowed.
    if (cookieSize != 0) {
      cgf.cgm.errorNYI(e->getSourceRange(),
                       "emitCXXNewAllocSize: array cookie");
    }

    size = cgf.getBuilder().getConstInt(loc, allocationSize);
  } else {
    // TODO: Handle the variable size case
    cgf.cgm.errorNYI(e->getSourceRange(),
                     "emitCXXNewAllocSize: variable array size");
  }

  if (cookieSize == 0)
    sizeWithoutCookie = size;
  else
    assert(sizeWithoutCookie && "didn't set sizeWithoutCookie?");

  return size;
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
    cgf.emitComplexExprIntoLValue(init, cgf.makeAddrLValue(newPtr, allocType),
                                  /*isInit*/ true);
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

void CIRGenFunction::emitNewArrayInitializer(
    const CXXNewExpr *e, QualType elementType, mlir::Type elementTy,
    Address beginPtr, mlir::Value numElements,
    mlir::Value allocSizeWithoutCookie) {
  // If we have a type with trivial initialization and no initializer,
  // there's nothing to do.
  if (!e->hasInitializer())
    return;

  cgm.errorNYI(e->getSourceRange(), "emitNewArrayInitializer");
}

static void emitNewInitializer(CIRGenFunction &cgf, const CXXNewExpr *e,
                               QualType elementType, mlir::Type elementTy,
                               Address newPtr, mlir::Value numElements,
                               mlir::Value allocSizeWithoutCookie) {
  assert(!cir::MissingFeatures::generateDebugInfo());
  if (e->isArray()) {
    cgf.emitNewArrayInitializer(e, elementType, elementTy, newPtr, numElements,
                                allocSizeWithoutCookie);
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

RValue CIRGenFunction::emitCXXPseudoDestructorExpr(
    const CXXPseudoDestructorExpr *expr) {
  QualType destroyedType = expr->getDestroyedType();
  if (destroyedType.hasStrongOrWeakObjCLifetime()) {
    assert(!cir::MissingFeatures::objCLifetime());
    cgm.errorNYI(expr->getExprLoc(),
                 "emitCXXPseudoDestructorExpr: Objective-C lifetime is NYI");
  } else {
    // C++ [expr.pseudo]p1:
    //   The result shall only be used as the operand for the function call
    //   operator (), and the result of such a call has type void. The only
    //   effect is the evaluation of the postfix-expression before the dot or
    //   arrow.
    emitIgnoredExpr(expr->getBase());
  }

  return RValue::get(nullptr);
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

namespace {
/// Calls the given 'operator delete' on a single object.
struct CallObjectDelete final : EHScopeStack::Cleanup {
  mlir::Value ptr;
  const FunctionDecl *operatorDelete;
  QualType elementType;

  CallObjectDelete(mlir::Value ptr, const FunctionDecl *operatorDelete,
                   QualType elementType)
      : ptr(ptr), operatorDelete(operatorDelete), elementType(elementType) {}

  void emit(CIRGenFunction &cgf) override {
    cgf.emitDeleteCall(operatorDelete, ptr, elementType);
  }
};
} // namespace

/// Emit the code for deleting a single object.
static void emitObjectDelete(CIRGenFunction &cgf, const CXXDeleteExpr *de,
                             Address ptr, QualType elementType) {
  // C++11 [expr.delete]p3:
  //   If the static type of the object to be deleted is different from its
  //   dynamic type, the static type shall be a base class of the dynamic type
  //   of the object to be deleted and the static type shall have a virtual
  //   destructor or the behavior is undefined.
  assert(!cir::MissingFeatures::emitTypeCheck());

  const FunctionDecl *operatorDelete = de->getOperatorDelete();
  assert(!operatorDelete->isDestroyingOperatorDelete());

  // Find the destructor for the type, if applicable.  If the
  // destructor is virtual, we'll just emit the vcall and return.
  const CXXDestructorDecl *dtor = nullptr;
  if (const auto *rd = elementType->getAsCXXRecordDecl()) {
    if (rd->hasDefinition() && !rd->hasTrivialDestructor()) {
      dtor = rd->getDestructor();

      if (dtor->isVirtual()) {
        cgf.cgm.errorNYI(de->getSourceRange(),
                         "emitObjectDelete: virtual destructor");
      }
    }
  }

  // Make sure that we call delete even if the dtor throws.
  // This doesn't have to a conditional cleanup because we're going
  // to pop it off in a second.
  cgf.ehStack.pushCleanup<CallObjectDelete>(
      NormalAndEHCleanup, ptr.getPointer(), operatorDelete, elementType);

  if (dtor) {
    cgf.emitCXXDestructorCall(dtor, Dtor_Complete,
                              /*ForVirtualBase=*/false,
                              /*Delegating=*/false, ptr, elementType);
  } else if (elementType.getObjCLifetime()) {
    assert(!cir::MissingFeatures::objCLifetime());
    cgf.cgm.errorNYI(de->getSourceRange(), "emitObjectDelete: ObjCLifetime");
  }

  // In traditional LLVM codegen null checks are emitted to save a delete call.
  // In CIR we optimize for size by default, the null check should be added into
  // this function callers.
  assert(!cir::MissingFeatures::emitNullCheckForDeleteCalls());

  cgf.popCleanupBlock();
}

void CIRGenFunction::emitCXXDeleteExpr(const CXXDeleteExpr *e) {
  const Expr *arg = e->getArgument();
  Address ptr = emitPointerWithAlignment(arg);

  // Null check the pointer.
  //
  // We could avoid this null check if we can determine that the object
  // destruction is trivial and doesn't require an array cookie; we can
  // unconditionally perform the operator delete call in that case. For now, we
  // assume that deleted pointers are null rarely enough that it's better to
  // keep the branch. This might be worth revisiting for a -O0 code size win.
  //
  // CIR note: emit the code size friendly by default for now, such as mentioned
  // in `emitObjectDelete`.
  assert(!cir::MissingFeatures::emitNullCheckForDeleteCalls());
  QualType deleteTy = e->getDestroyedType();

  // A destroying operator delete overrides the entire operation of the
  // delete expression.
  if (e->getOperatorDelete()->isDestroyingOperatorDelete()) {
    cgm.errorNYI(e->getSourceRange(),
                 "emitCXXDeleteExpr: destroying operator delete");
    return;
  }

  // We might be deleting a pointer to array.
  deleteTy = getContext().getBaseElementType(deleteTy);
  ptr = ptr.withElementType(builder, convertTypeForMem(deleteTy));

  if (e->isArrayForm()) {
    assert(!cir::MissingFeatures::deleteArray());
    cgm.errorNYI(e->getSourceRange(), "emitCXXDeleteExpr: array delete");
    return;
  } else {
    emitObjectDelete(*this, e, ptr, deleteTy);
  }
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

  mlir::Type elementTy;
  if (e->isArray()) {
    // For array new, use the allocated type to handle multidimensional arrays
    // correctly
    elementTy = convertTypeForMem(e->getAllocatedType());
  } else {
    elementTy = convertTypeForMem(allocType);
  }
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

void CIRGenFunction::emitDeleteCall(const FunctionDecl *deleteFD,
                                    mlir::Value ptr, QualType deleteTy) {
  assert(!cir::MissingFeatures::deleteArray());

  const auto *deleteFTy = deleteFD->getType()->castAs<FunctionProtoType>();
  CallArgList deleteArgs;

  UsualDeleteParams params = deleteFD->getUsualDeleteParams();
  auto paramTypeIt = deleteFTy->param_type_begin();

  // Pass std::type_identity tag if present
  if (isTypeAwareAllocation(params.TypeAwareDelete))
    cgm.errorNYI(deleteFD->getSourceRange(),
                 "emitDeleteCall: type aware delete");

  // Pass the pointer itself.
  QualType argTy = *paramTypeIt++;
  mlir::Value deletePtr =
      builder.createBitcast(ptr.getLoc(), ptr, convertType(argTy));
  deleteArgs.add(RValue::get(deletePtr), argTy);

  // Pass the std::destroying_delete tag if present.
  if (params.DestroyingDelete)
    cgm.errorNYI(deleteFD->getSourceRange(),
                 "emitDeleteCall: destroying delete");

  // Pass the size if the delete function has a size_t parameter.
  if (params.Size) {
    QualType sizeType = *paramTypeIt++;
    CharUnits deleteTypeSize = getContext().getTypeSizeInChars(deleteTy);
    assert(mlir::isa<cir::IntType>(convertType(sizeType)) &&
           "expected cir::IntType");
    cir::ConstantOp size = builder.getConstInt(
        *currSrcLoc, convertType(sizeType), deleteTypeSize.getQuantity());

    deleteArgs.add(RValue::get(size), sizeType);
  }

  // Pass the alignment if the delete function has an align_val_t parameter.
  if (isAlignedAllocation(params.Alignment))
    cgm.errorNYI(deleteFD->getSourceRange(),
                 "emitDeleteCall: aligned allocation");

  assert(paramTypeIt == deleteFTy->param_type_end() &&
         "unknown parameter to usual delete function");

  // Emit the call to delete.
  emitNewDeleteCall(*this, deleteFD, deleteFTy, deleteArgs);
}
