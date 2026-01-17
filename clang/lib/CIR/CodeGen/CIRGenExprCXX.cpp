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
#include "clang/Basic/OperatorKinds.h"
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

RValue
CIRGenFunction::emitCXXMemberPointerCallExpr(const CXXMemberCallExpr *ce,
                                             ReturnValueSlot returnValue) {
  const BinaryOperator *bo =
      cast<BinaryOperator>(ce->getCallee()->IgnoreParens());
  const Expr *baseExpr = bo->getLHS();
  const Expr *memFnExpr = bo->getRHS();

  const auto *mpt = memFnExpr->getType()->castAs<MemberPointerType>();
  const auto *fpt = mpt->getPointeeType()->castAs<FunctionProtoType>();

  // Emit the 'this' pointer.
  Address thisAddr = Address::invalid();
  if (bo->getOpcode() == BO_PtrMemI)
    thisAddr = emitPointerWithAlignment(baseExpr);
  else
    thisAddr = emitLValue(baseExpr).getAddress();

  assert(!cir::MissingFeatures::emitTypeCheck());

  // Get the member function pointer.
  mlir::Value memFnPtr = emitScalarExpr(memFnExpr);

  // Resolve the member function pointer to the actual callee and adjust the
  // "this" pointer for call.
  mlir::Location loc = getLoc(ce->getExprLoc());
  auto [/*mlir::Value*/ calleePtr, /*mlir::Value*/ adjustedThis] =
      builder.createGetMethod(loc, memFnPtr, thisAddr.getPointer());

  // Prepare the call arguments.
  CallArgList argsList;
  argsList.add(RValue::get(adjustedThis), getContext().VoidPtrTy);
  emitCallArgs(argsList, fpt, ce->arguments());

  RequiredArgs required = RequiredArgs::getFromProtoWithExtraSlots(fpt, 1);

  // Build the call.
  CIRGenCallee callee(fpt, calleePtr.getDefiningOp());
  assert(!cir::MissingFeatures::opCallMustTail());
  return emitCall(cgm.getTypes().arrangeCXXMethodCall(argsList, fpt, required,
                                                      /*PrefixSize=*/0),
                  callee, returnValue, argsList, nullptr, loc);
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
  if (const auto *dtor = dyn_cast<CXXDestructorDecl>(calleeDecl))
    fInfo = &cgm.getTypes().arrangeCXXStructorDeclaration(
        GlobalDecl(dtor, Dtor_Complete));
  else
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

  if (const auto *dtor = dyn_cast<CXXDestructorDecl>(calleeDecl)) {
    assert(ce->arg_begin() == ce->arg_end() &&
           "Destructor shouldn't have explicit parameters");
    assert(returnValue.isNull() && "Destructor shouldn't have return value");
    if (useVirtualCall) {
      cgm.getCXXABI().emitVirtualDestructorCall(*this, dtor, Dtor_Complete,
                                                thisPtr.getAddress(),
                                                cast<CXXMemberCallExpr>(ce));
    } else {
      GlobalDecl globalDecl(dtor, Dtor_Complete);
      CIRGenCallee callee;
      assert(!cir::MissingFeatures::appleKext());
      if (!devirtualizedMethod) {
        callee = CIRGenCallee::forDirect(
            cgm.getAddrOfCXXStructor(globalDecl, fInfo, ty), globalDecl);
      } else {
        cgm.errorNYI(ce->getSourceRange(), "devirtualized destructor call");
        return RValue::get(nullptr);
      }

      QualType thisTy =
          isArrow ? base->getType()->getPointeeType() : base->getType();
      // CIRGen does not pass CallOrInvoke here (different from OG LLVM codegen)
      // because in practice it always null even in OG.
      emitCXXDestructorCall(globalDecl, callee, thisPtr.getPointer(), thisTy,
                            /*implicitParam=*/nullptr,
                            /*implicitParamTy=*/QualType(), ce);
    }
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

static void emitNullBaseClassInitialization(CIRGenFunction &cgf,
                                            Address destPtr,
                                            const CXXRecordDecl *base) {
  if (base->isEmpty())
    return;

  const ASTRecordLayout &layout = cgf.getContext().getASTRecordLayout(base);
  CharUnits nvSize = layout.getNonVirtualSize();

  // We cannot simply zero-initialize the entire base sub-object if vbptrs are
  // present, they are initialized by the most derived class before calling the
  // constructor.
  SmallVector<std::pair<CharUnits, CharUnits>, 1> stores;
  stores.emplace_back(CharUnits::Zero(), nvSize);

  // Each store is split by the existence of a vbptr.
  // TODO(cir): This only needs handling for the MS CXXABI.
  assert(!cir::MissingFeatures::msabi());

  // If the type contains a pointer to data member we can't memset it to zero.
  // Instead, create a null constant and copy it to the destination.
  // TODO: there are other patterns besides zero that we can usefully memset,
  // like -1, which happens to be the pattern used by member-pointers.
  // TODO: isZeroInitializable can be over-conservative in the case where a
  // virtual base contains a member pointer.
  mlir::TypedAttr nullConstantForBase = cgf.cgm.emitNullConstantForBase(base);
  if (!cgf.getBuilder().isNullValue(nullConstantForBase)) {
    cgf.cgm.errorNYI(
        base->getSourceRange(),
        "emitNullBaseClassInitialization: base constant is not null");
  } else {
    // Otherwise, just memset the whole thing to zero.  This is legal
    // because in LLVM, all default initializers (other than the ones we just
    // handled above) are guaranteed to have a bit pattern of all zeros.
    // TODO(cir): When the MS CXXABI is supported, we will need to iterate over
    // `stores` and create a separate memset for each one. For now, we know that
    // there will only be one store and it will begin at offset zero, so that
    // simplifies this code considerably.
    assert(stores.size() == 1 && "Expected only one store");
    assert(stores[0].first == CharUnits::Zero() &&
           "Expected store to begin at offset zero");
    CIRGenBuilderTy builder = cgf.getBuilder();
    mlir::Location loc = cgf.getLoc(base->getBeginLoc());
    builder.createStore(loc, builder.getConstant(loc, nullConstantForBase),
                        destPtr);
  }
}

void CIRGenFunction::emitCXXConstructExpr(const CXXConstructExpr *e,
                                          AggValueSlot dest) {
  assert(!dest.isIgnored() && "Must have a destination!");
  const CXXConstructorDecl *cd = e->getConstructor();

  // If we require zero initialization before (or instead of) calling the
  // constructor, as can be the case with a non-user-provided default
  // constructor, emit the zero initialization now, unless destination is
  // already zeroed.
  if (e->requiresZeroInitialization() && !dest.isZeroed()) {
    switch (e->getConstructionKind()) {
    case CXXConstructionKind::Delegating:
    case CXXConstructionKind::Complete:
      emitNullInitialization(getLoc(e->getSourceRange()), dest.getAddress(),
                             e->getType());
      break;
    case CXXConstructionKind::VirtualBase:
    case CXXConstructionKind::NonVirtualBase:
      emitNullBaseClassInitialization(*this, dest.getAddress(),
                                      cd->getParent());
      break;
    }
  }

  // If this is a call to a trivial default constructor, do nothing.
  if (cd->isTrivial() && cd->isDefaultConstructor())
    return;

  // Elide the constructor if we're constructing from a temporary
  if (getLangOpts().ElideConstructors && e->isElidable()) {
    // FIXME: This only handles the simplest case, where the source object is
    //        passed directly as the first argument to the constructor. This
    //        should also handle stepping through implicit casts and conversion
    //        sequences which involve two steps, with a conversion operator
    //        follwed by a converting constructor.
    const Expr *srcObj = e->getArg(0);
    assert(srcObj->isTemporaryObject(getContext(), cd->getParent()));
    assert(
        getContext().hasSameUnqualifiedType(e->getType(), srcObj->getType()));
    emitAggExpr(srcObj, dest);
    return;
  }

  if (const ArrayType *arrayType = getContext().getAsArrayType(e->getType())) {
    assert(!cir::MissingFeatures::sanitizers());
    emitCXXAggrConstructorCall(cd, arrayType, dest.getAddress(), e, false);
  } else {

    clang::CXXCtorType type = Ctor_Complete;
    bool forVirtualBase = false;
    bool delegating = false;

    switch (e->getConstructionKind()) {
    case CXXConstructionKind::Complete:
      type = Ctor_Complete;
      break;
    case CXXConstructionKind::Delegating:
      // We should be emitting a constructor; GlobalDecl will assert this
      type = curGD.getCtorType();
      delegating = true;
      break;
    case CXXConstructionKind::VirtualBase:
      forVirtualBase = true;
      [[fallthrough]];
    case CXXConstructionKind::NonVirtualBase:
      type = Ctor_Base;
      break;
    }

    emitCXXConstructorCall(cd, type, forVirtualBase, delegating, dest, e);
  }
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
        loc, cir::IntAttr::get(cgf.sizeTy, typeSize.getQuantity()));
    return sizeWithoutCookie;
  }

  // The width of size_t.
  unsigned sizeWidth = cgf.cgm.getDataLayout().getTypeSizeInBits(cgf.sizeTy);

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

    [[maybe_unused]] unsigned numElementsWidth = count.getBitWidth();
    bool hasAnyOverflow = false;

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
      // Save the current size without a cookie.  This shouldn't be
      // used if there was overflow
      sizeWithoutCookie = cgf.getBuilder().getConstInt(
          loc, allocationSize.zextOrTrunc(sizeWidth));

      allocationSize = allocationSize.uadd_ov(cookieSize, overflow);
      hasAnyOverflow |= overflow;
    }

    // On overflow, produce a -1 so operator new will fail
    if (hasAnyOverflow) {
      size =
          cgf.getBuilder().getConstInt(loc, llvm::APInt::getAllOnes(sizeWidth));
    } else {
      size = cgf.getBuilder().getConstInt(loc, allocationSize);
    }
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

  unsigned initListElements = 0;

  const Expr *init = e->getInitializer();
  const InitListExpr *ile = dyn_cast<InitListExpr>(init);
  if (ile) {
    cgm.errorNYI(ile->getSourceRange(), "emitNewArrayInitializer: init list");
    return;
  }

  // If all elements have already been initialized, skip any further
  // initialization.
  auto constOp = mlir::dyn_cast<cir::ConstantOp>(numElements.getDefiningOp());
  if (constOp) {
    auto constIntAttr = mlir::dyn_cast<cir::IntAttr>(constOp.getValue());
    // Just skip out if the constant count is zero.
    if (constIntAttr && constIntAttr.getUInt() <= initListElements)
      return;
  }

  assert(init && "have trailing elements to initialize but no initializer");

  // If this is a constructor call, try to optimize it out, and failing that
  // emit a single loop to initialize all remaining elements.
  if (const CXXConstructExpr *cce = dyn_cast<CXXConstructExpr>(init)) {
    CXXConstructorDecl *ctor = cce->getConstructor();
    if (ctor->isTrivial()) {
      // If new expression did not specify value-initialization, then there
      // is no initialization.
      if (!cce->requiresZeroInitialization())
        return;

      cgm.errorNYI(cce->getSourceRange(),
                   "emitNewArrayInitializer: trivial ctor zero-init");
      return;
    }

    cgm.errorNYI(cce->getSourceRange(),
                 "emitNewArrayInitializer: ctor initializer");
    return;
  }

  cgm.errorNYI(init->getSourceRange(),
               "emitNewArrayInitializer: unsupported initializer");
  return;
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

RValue CIRGenFunction::emitNewOrDeleteBuiltinCall(const FunctionProtoType *type,
                                                  const CallExpr *callExpr,
                                                  OverloadedOperatorKind op) {
  CallArgList args;
  emitCallArgs(args, type, callExpr->arguments());
  // Find the allocation or deallocation function that we're calling.
  ASTContext &astContext = getContext();
  assert(op == OO_New || op == OO_Delete);
  DeclarationName name = astContext.DeclarationNames.getCXXOperatorName(op);

  clang::DeclContextLookupResult lookupResult =
      astContext.getTranslationUnitDecl()->lookup(name);
  for (const auto *decl : lookupResult) {
    if (const auto *funcDecl = dyn_cast<FunctionDecl>(decl)) {
      if (astContext.hasSameType(funcDecl->getType(), QualType(type, 0))) {
        if (sanOpts.has(SanitizerKind::AllocToken)) {
          // TODO: Set !alloc_token metadata.
          assert(!cir::MissingFeatures::allocToken());
          cgm.errorNYI("Alloc token sanitizer not yet supported!");
        }

        // Emit the call to operator new/delete.
        return emitNewDeleteCall(*this, funcDecl, type, args);
      }
    }
  }

  llvm_unreachable("predeclared global operator new/delete is missing");
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

  void emit(CIRGenFunction &cgf, Flags flags) override {
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
        assert(!cir::MissingFeatures::devirtualizeDestructor());
        cgf.cgm.getCXXABI().emitVirtualObjectDelete(cgf, de, ptr, elementType,
                                                    dtor);
        return;
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

  if (e->isArrayForm() &&
      cgm.getASTContext().getTargetInfo().emitVectorDeletingDtors(
          cgm.getASTContext().getLangOpts())) {
    cgm.errorNYI(e->getSourceRange(),
                 "emitCXXDeleteExpr: emitVectorDeletingDtors");
  }

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

  mlir::Value numElements = nullptr;
  mlir::Value allocSizeWithoutCookie = nullptr;
  mlir::Value allocSize = emitCXXNewAllocSize(
      *this, e, minElements, numElements, allocSizeWithoutCookie);
  CharUnits allocAlign = getContext().getTypeAlignInChars(allocType);

  // Emit the allocation call.
  Address allocation = Address::invalid();
  CallArgList allocatorArgs;
  if (allocator->isReservedGlobalPlacementOperator()) {
    // If the allocator is a global placement operator, just
    // "inline" it directly.
    assert(e->getNumPlacementArgs() == 1);
    const Expr *arg = *e->placement_arguments().begin();

    LValueBaseInfo baseInfo;
    allocation = emitPointerWithAlignment(arg, &baseInfo);

    // The pointer expression will, in many cases, be an opaque void*.
    // In these cases, discard the computed alignment and use the
    // formal alignment of the allocated type.
    if (baseInfo.getAlignmentSource() != AlignmentSource::Decl)
      allocation = allocation.withAlignment(allocAlign);

    // Set up allocatorArgs for the call to operator delete if it's not
    // the reserved global operator.
    if (e->getOperatorDelete() &&
        !e->getOperatorDelete()->isReservedGlobalPlacementOperator()) {
      cgm.errorNYI(e->getSourceRange(),
                   "emitCXXNewExpr: reserved placement new with delete");
    }
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

  if (allocSize != allocSizeWithoutCookie) {
    assert(e->isArray());
    allocation = cgm.getCXXABI().initializeArrayCookie(
        *this, allocation, numElements, e, allocType);
  }

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

static mlir::Value emitDynamicCastToNull(CIRGenFunction &cgf,
                                         mlir::Location loc, QualType destTy) {
  mlir::Type destCIRTy = cgf.convertType(destTy);
  assert(mlir::isa<cir::PointerType>(destCIRTy) &&
         "result of dynamic_cast should be a ptr");

  if (!destTy->isPointerType()) {
    mlir::Region *currentRegion = cgf.getBuilder().getBlock()->getParent();
    /// C++ [expr.dynamic.cast]p9:
    ///   A failed cast to reference type throws std::bad_cast
    cgf.cgm.getCXXABI().emitBadCastCall(cgf, loc);

    // The call to bad_cast will terminate the current block. Create a new block
    // to hold any follow up code.
    cgf.getBuilder().createBlock(currentRegion, currentRegion->end());
  }

  return cgf.getBuilder().getNullPtr(destCIRTy, loc);
}

mlir::Value CIRGenFunction::emitDynamicCast(Address thisAddr,
                                            const CXXDynamicCastExpr *dce) {
  mlir::Location loc = getLoc(dce->getSourceRange());

  cgm.emitExplicitCastExprType(dce, this);
  QualType destTy = dce->getTypeAsWritten();
  QualType srcTy = dce->getSubExpr()->getType();

  // C++ [expr.dynamic.cast]p7:
  //   If T is "pointer to cv void," then the result is a pointer to the most
  //   derived object pointed to by v.
  bool isDynCastToVoid = destTy->isVoidPointerType();
  bool isRefCast = destTy->isReferenceType();

  QualType srcRecordTy;
  QualType destRecordTy;
  if (isDynCastToVoid) {
    srcRecordTy = srcTy->getPointeeType();
    // No destRecordTy.
  } else if (const PointerType *destPTy = destTy->getAs<PointerType>()) {
    srcRecordTy = srcTy->castAs<PointerType>()->getPointeeType();
    destRecordTy = destPTy->getPointeeType();
  } else {
    srcRecordTy = srcTy;
    destRecordTy = destTy->castAs<ReferenceType>()->getPointeeType();
  }

  assert(srcRecordTy->isRecordType() && "source type must be a record type!");
  assert(!cir::MissingFeatures::emitTypeCheck());

  if (dce->isAlwaysNull())
    return emitDynamicCastToNull(*this, loc, destTy);

  auto destCirTy = mlir::cast<cir::PointerType>(convertType(destTy));
  return cgm.getCXXABI().emitDynamicCast(*this, loc, srcRecordTy, destRecordTy,
                                         destCirTy, isRefCast, thisAddr);
}
