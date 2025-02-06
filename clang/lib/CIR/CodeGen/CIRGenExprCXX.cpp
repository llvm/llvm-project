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

#include "clang/AST/CharUnits.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/MissingFeatures.h"
#include <CIRGenCXXABI.h>
#include <CIRGenCstEmitter.h>
#include <CIRGenFunction.h>
#include <CIRGenModule.h>
#include <CIRGenValue.h>

#include <clang/AST/DeclCXX.h>

using namespace clang;
using namespace clang::CIRGen;

namespace {
struct MemberCallInfo {
  RequiredArgs ReqArgs;
  // Number of prefix arguments for the call. Ignores the `this` pointer.
  unsigned PrefixSize;
};
} // namespace

static RValue emitNewDeleteCall(CIRGenFunction &CGF,
                                const FunctionDecl *CalleeDecl,
                                const FunctionProtoType *CalleeType,
                                const CallArgList &Args);

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
    Args.add(RValue::get(ImplicitParam), ImplicitParamTy);
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
    CGF.emitCallArgs(Args, FPT, drop_begin(CE->arguments(), ArgsToSkip),
                     CE->getDirectCallee());
  } else {
    assert(
        FPT->getNumParams() == 0 &&
        "No CallExpr specified for function with non-zero number of arguments");
  }

  return {required, PrefixSize};
}

RValue CIRGenFunction::emitCXXMemberOrOperatorCall(
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
  return emitCall(FnInfo, Callee, ReturnValue, Args, nullptr,
                  CE && CE == MustTailCall, loc, CE);
}

// TODO(cir): this can be shared with LLVM codegen
static CXXRecordDecl *getCXXRecord(const Expr *E) {
  QualType T = E->getType();
  if (const PointerType *PTy = T->getAs<PointerType>())
    T = PTy->getPointeeType();
  const RecordType *Ty = T->castAs<RecordType>();
  return cast<CXXRecordDecl>(Ty->getDecl());
}

RValue
CIRGenFunction::emitCXXMemberPointerCallExpr(const CXXMemberCallExpr *E,
                                             ReturnValueSlot ReturnValue) {
  const BinaryOperator *BO =
      cast<BinaryOperator>(E->getCallee()->IgnoreParens());
  const Expr *BaseExpr = BO->getLHS();
  const Expr *MemFnExpr = BO->getRHS();

  const auto *MPT = MemFnExpr->getType()->castAs<MemberPointerType>();
  const auto *FPT = MPT->getPointeeType()->castAs<FunctionProtoType>();

  // Emit the 'this' pointer.
  Address This = Address::invalid();
  if (BO->getOpcode() == BO_PtrMemI)
    This = emitPointerWithAlignment(BaseExpr, nullptr, nullptr, KnownNonNull);
  else
    This = emitLValue(BaseExpr).getAddress();

  emitTypeCheck(TCK_MemberCall, E->getExprLoc(), This.emitRawPointer(),
                QualType(MPT->getClass(), 0));

  // Get the member function pointer.
  mlir::Value MemFnPtr = emitScalarExpr(MemFnExpr);

  // Resolve the member function pointer to the actual callee and adjust the
  // "this" pointer for call.
  auto Loc = getLoc(E->getExprLoc());
  auto [CalleePtr, AdjustedThis] =
      builder.createGetMethod(Loc, MemFnPtr, This.getPointer());

  // Prepare the call arguments.
  CallArgList ArgsList;
  ArgsList.add(RValue::get(AdjustedThis), getContext().VoidPtrTy);
  emitCallArgs(ArgsList, FPT, E->arguments());

  RequiredArgs required = RequiredArgs::forPrototypePlus(FPT, 1);

  // Build the call.
  CIRGenCallee Callee(FPT, CalleePtr.getDefiningOp());
  return emitCall(CGM.getTypes().arrangeCXXMethodCall(ArgsList, FPT, required,
                                                      /*PrefixSize=*/0),
                  Callee, ReturnValue, ArgsList, nullptr, E == MustTailCall,
                  Loc);
}

RValue CIRGenFunction::emitCXXMemberOrOperatorMemberCallExpr(
    const CallExpr *CE, const CXXMethodDecl *MD, ReturnValueSlot ReturnValue,
    bool HasQualifier, NestedNameSpecifier *Qualifier, bool IsArrow,
    const Expr *Base) {
  assert(isa<CXXMemberCallExpr>(CE) || isa<CXXOperatorCallExpr>(CE));

  // Compute the object pointer.
  bool CanUseVirtualCall = MD->isVirtual() && !HasQualifier;
  const CXXMethodDecl *DevirtualizedMethod = nullptr;
  if (CanUseVirtualCall &&
      MD->getDevirtualizedMethod(Base, getLangOpts().AppleKext)) {
    const CXXRecordDecl *BestDynamicDecl = Base->getBestDynamicClassType();
    DevirtualizedMethod = MD->getCorrespondingMethodInClass(BestDynamicDecl);
    assert(DevirtualizedMethod);
    const CXXRecordDecl *DevirtualizedClass = DevirtualizedMethod->getParent();
    const Expr *Inner = Base->IgnoreParenBaseCasts();
    if (DevirtualizedMethod->getReturnType().getCanonicalType() !=
        MD->getReturnType().getCanonicalType()) {
      // If the return types are not the same, this might be a case where more
      // code needs to run to compensate for it. For example, the derived
      // method might return a type that inherits form from the return
      // type of MD and has a prefix.
      // For now we just avoid devirtualizing these covariant cases.
      DevirtualizedMethod = nullptr;
    } else if (getCXXRecord(Inner) == DevirtualizedClass) {
      // If the class of the Inner expression is where the dynamic method
      // is defined, build the this pointer from it.
      Base = Inner;
    } else if (getCXXRecord(Base) != DevirtualizedClass) {
      // If the method is defined in a class that is not the best dynamic
      // one or the one of the full expression, we would have to build
      // a derived-to-base cast to compute the correct this pointer, but
      // we don't have support for that yet, so do a virtual call.
      assert(!cir::MissingFeatures::emitDerivedToBaseCastForDevirt());
      DevirtualizedMethod = nullptr;
    }
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
      //    TrivialAssignmentRHS = emitLValue(CE->getArg(1));
      //  } else {
      RtlArgs = &RtlArgStorage;
      emitCallArgs(*RtlArgs, MD->getType()->castAs<FunctionProtoType>(),
                   drop_begin(CE->arguments(), 1), CE->getDirectCallee(),
                   /*ParamsToSkip*/ 0, EvaluationOrder::ForceRightToLeft);
    }
  }

  LValue This;
  if (IsArrow) {
    LValueBaseInfo BaseInfo;
    TBAAAccessInfo TBAAInfo;
    Address ThisValue = emitPointerWithAlignment(Base, &BaseInfo, &TBAAInfo);
    This = makeAddrLValue(ThisValue, Base->getType(), BaseInfo, TBAAInfo);
  } else {
    This = emitLValue(Base);
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
      //                                           emitLValue(*CE->arg_begin());
      // emitAggregateAssign(This, RHS, CE->getType());
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
    FInfo = &CGM.getTypes().arrangeCXXStructorDeclaration(
        GlobalDecl(Dtor, Dtor_Complete));
  else
    FInfo = &CGM.getTypes().arrangeCXXMethodDeclaration(CalleeDecl);

  auto Ty = CGM.getTypes().GetFunctionType(*FInfo);

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

  if (cir::MissingFeatures::emitTypeCheck())
    llvm_unreachable("NYI");

  // C++ [class.virtual]p12:
  //   Explicit qualification with the scope operator (5.1) suppresses the
  //   virtual call mechanism.
  //
  // We also don't emit a virtual call if the base expression has a record type
  // because then we know what the type is.
  bool useVirtualCall = CanUseVirtualCall && !DevirtualizedMethod;

  if (const auto *dtor = dyn_cast<CXXDestructorDecl>(CalleeDecl)) {
    assert(CE->arg_begin() == CE->arg_end() &&
           "Destructor shouldn't have explicit parameters");
    assert(ReturnValue.isNull() && "Destructor shouldn't have return value");
    if (useVirtualCall) {
      llvm_unreachable("NYI");
    } else {
      GlobalDecl globalDecl(dtor, Dtor_Complete);
      CIRGenCallee Callee;
      if (getLangOpts().AppleKext && dtor->isVirtual() && HasQualifier)
        llvm_unreachable("NYI");
      else if (!DevirtualizedMethod)
        Callee = CIRGenCallee::forDirect(
            CGM.getAddrOfCXXStructor(globalDecl, FInfo, Ty), globalDecl);
      else {
        Callee = CIRGenCallee::forDirect(CGM.GetAddrOfFunction(globalDecl, Ty),
                                         globalDecl);
      }

      QualType thisTy =
          IsArrow ? Base->getType()->getPointeeType() : Base->getType();
      // CIRGen does not pass CallOrInvoke here (different from OG LLVM codegen)
      // because in practice it always null even in OG.
      emitCXXDestructorCall(globalDecl, Callee, This.getPointer(), thisTy,
                            /*ImplicitParam=*/nullptr,
                            /*ImplicitParamTy=*/QualType(), CE);
    }
    return RValue::get(nullptr);
  }

  // FIXME: Uses of 'MD' past this point need to be audited. We may need to use
  // 'CalleeDecl' instead.

  CIRGenCallee Callee;
  if (useVirtualCall) {
    Callee = CIRGenCallee::forVirtual(CE, MD, This.getAddress(), Ty);
  } else {
    if (SanOpts.has(SanitizerKind::CFINVCall)) {
      llvm_unreachable("NYI");
    }

    if (getLangOpts().AppleKext)
      llvm_unreachable("NYI");
    else if (!DevirtualizedMethod)
      // TODO(cir): shouldn't this call getAddrOfCXXStructor instead?
      Callee = CIRGenCallee::forDirect(CGM.GetAddrOfFunction(MD, Ty),
                                       GlobalDecl(MD));
    else {
      Callee = CIRGenCallee::forDirect(CGM.GetAddrOfFunction(MD, Ty),
                                       GlobalDecl(MD));
    }
  }

  if (MD->isVirtual()) {
    Address NewThisAddr =
        CGM.getCXXABI().adjustThisArgumentForVirtualFunctionCall(
            *this, CalleeDecl, This.getAddress(), useVirtualCall);
    This.setAddress(NewThisAddr);
  }

  return emitCXXMemberOrOperatorCall(
      CalleeDecl, Callee, ReturnValue, This.getPointer(),
      /*ImplicitParam=*/nullptr, QualType(), CE, RtlArgs);
}

RValue
CIRGenFunction::emitCXXOperatorMemberCallExpr(const CXXOperatorCallExpr *E,
                                              const CXXMethodDecl *MD,
                                              ReturnValueSlot ReturnValue) {
  assert(MD->isInstance() &&
         "Trying to emit a member call expr on a static method!");
  return emitCXXMemberOrOperatorMemberCallExpr(
      E, MD, ReturnValue, /*HasQualifier=*/false, /*Qualifier=*/nullptr,
      /*IsArrow=*/false, E->getArg(0));
}

static void emitNullBaseClassInitialization(CIRGenFunction &CGF,
                                            Address DestPtr,
                                            const CXXRecordDecl *Base) {
  if (Base->isEmpty())
    return;

  DestPtr = DestPtr.withElementType(CGF.UInt8Ty);

  const ASTRecordLayout &Layout = CGF.getContext().getASTRecordLayout(Base);
  CharUnits NVSize = Layout.getNonVirtualSize();

  // We cannot simply zero-initialize the entire base sub-object if vbptrs are
  // present, they are initialized by the most derived class before calling the
  // constructor.
  SmallVector<std::pair<CharUnits, CharUnits>, 1> Stores;
  Stores.emplace_back(CharUnits::Zero(), NVSize);

  // Each store is split by the existence of a vbptr.
  CharUnits VBPtrWidth = CGF.getPointerSize();
  std::vector<CharUnits> VBPtrOffsets =
      CGF.CGM.getCXXABI().getVBPtrOffsets(Base);
  for (CharUnits VBPtrOffset : VBPtrOffsets) {
    // Stop before we hit any virtual base pointers located in virtual bases.
    if (VBPtrOffset >= NVSize)
      break;
    std::pair<CharUnits, CharUnits> LastStore = Stores.pop_back_val();
    CharUnits LastStoreOffset = LastStore.first;
    CharUnits LastStoreSize = LastStore.second;

    CharUnits SplitBeforeOffset = LastStoreOffset;
    CharUnits SplitBeforeSize = VBPtrOffset - SplitBeforeOffset;
    assert(!SplitBeforeSize.isNegative() && "negative store size!");
    if (!SplitBeforeSize.isZero())
      Stores.emplace_back(SplitBeforeOffset, SplitBeforeSize);

    CharUnits SplitAfterOffset = VBPtrOffset + VBPtrWidth;
    CharUnits SplitAfterSize = LastStoreSize - SplitAfterOffset;
    assert(!SplitAfterSize.isNegative() && "negative store size!");
    if (!SplitAfterSize.isZero())
      Stores.emplace_back(SplitAfterOffset, SplitAfterSize);
  }

  // If the type contains a pointer to data member we can't memset it to zero.
  // Instead, create a null constant and copy it to the destination.
  // TODO: there are other patterns besides zero that we can usefully memset,
  // like -1, which happens to be the pattern used by member-pointers.
  // TODO: isZeroInitializable can be over-conservative in the case where a
  // virtual base contains a member pointer.
  // TODO(cir): `nullConstantForBase` might be better off as a value instead
  // of an mlir::TypedAttr? Once this moves out of skeleton, make sure to double
  // check on what's better.
  mlir::Attribute nullConstantForBase = CGF.CGM.emitNullConstantForBase(Base);
  if (!CGF.getBuilder().isNullValue(nullConstantForBase)) {
    llvm_unreachable("NYI");
    // Otherwise, just memset the whole thing to zero.  This is legal
    // because in LLVM, all default initializers (other than the ones we just
    // handled above) are guaranteed to have a bit pattern of all zeros.
  } else {
    llvm_unreachable("NYI");
  }
}

void CIRGenFunction::emitCXXConstructExpr(const CXXConstructExpr *E,
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
      emitNullInitialization(getLoc(E->getSourceRange()), Dest.getAddress(),
                             E->getType());
      break;
    case CXXConstructionKind::VirtualBase:
    case CXXConstructionKind::NonVirtualBase:
      emitNullBaseClassInitialization(*this, Dest.getAddress(),
                                      CD->getParent());
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
    emitAggExpr(SrcObj, Dest);
    return;
  }

  if (const ArrayType *arrayType = getContext().getAsArrayType(E->getType())) {
    emitCXXAggrConstructorCall(CD, arrayType, Dest.getAddress(), E,
                               Dest.isSanitizerChecked());
  } else {
    clang::CXXCtorType Type = Ctor_Complete;
    bool ForVirtualBase = false;
    bool Delegating = false;

    switch (E->getConstructionKind()) {
    case CXXConstructionKind::Complete:
      Type = Ctor_Complete;
      break;
    case CXXConstructionKind::Delegating:
      // We should be emitting a constructor; GlobalDecl will assert this
      Type = CurGD.getCtorType();
      Delegating = true;
      break;
    case CXXConstructionKind::VirtualBase:
      ForVirtualBase = true;
      [[fallthrough]];
    case CXXConstructionKind::NonVirtualBase:
      Type = Ctor_Base;
      break;
    }

    emitCXXConstructorCall(CD, Type, ForVirtualBase, Delegating, Dest, E);
  }
}

namespace {
/// The parameters to pass to a usual operator delete.
struct UsualDeleteParams {
  bool DestroyingDelete = false;
  bool Size = false;
  bool Alignment = false;
};
} // namespace

// FIXME(cir): this should be shared with LLVM codegen
static UsualDeleteParams getUsualDeleteParams(const FunctionDecl *FD) {
  UsualDeleteParams Params;

  const FunctionProtoType *FPT = FD->getType()->castAs<FunctionProtoType>();
  auto AI = FPT->param_type_begin(), AE = FPT->param_type_end();

  // The first argument is always a void*.
  ++AI;

  // The next parameter may be a std::destroying_delete_t.
  if (FD->isDestroyingOperatorDelete()) {
    Params.DestroyingDelete = true;
    assert(AI != AE);
    ++AI;
  }

  // Figure out what other parameters we should be implicitly passing.
  if (AI != AE && (*AI)->isIntegerType()) {
    Params.Size = true;
    ++AI;
  }

  if (AI != AE && (*AI)->isAlignValT()) {
    Params.Alignment = true;
    ++AI;
  }

  assert(AI == AE && "unexpected usual deallocation function parameter");
  return Params;
}

static CharUnits CalculateCookiePadding(CIRGenFunction &CGF,
                                        const CXXNewExpr *E) {
  if (!E->isArray())
    return CharUnits::Zero();

  // No cookie is required if the operator new[] being used is the
  // reserved placement operator new[].
  if (E->getOperatorNew()->isReservedGlobalPlacementOperator())
    return CharUnits::Zero();

  return CGF.CGM.getCXXABI().getArrayCookieSize(E);
}

static mlir::Value emitCXXNewAllocSize(CIRGenFunction &CGF, const CXXNewExpr *e,
                                       unsigned minElements,
                                       mlir::Value &numElements,
                                       mlir::Value &sizeWithoutCookie) {
  QualType type = e->getAllocatedType();
  mlir::Location Loc = CGF.getLoc(e->getSourceRange());

  if (!e->isArray()) {
    CharUnits typeSize = CGF.getContext().getTypeSizeInChars(type);
    sizeWithoutCookie = CGF.getBuilder().getConstant(
        CGF.getLoc(e->getSourceRange()),
        cir::IntAttr::get(CGF.SizeTy, typeSize.getQuantity()));
    return sizeWithoutCookie;
  }

  // The width of size_t.
  unsigned sizeWidth = CGF.CGM.getDataLayout().getTypeSizeInBits(CGF.SizeTy);

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
  while (const ConstantArrayType *CAT =
             CGF.getContext().getAsConstantArrayType(type)) {
    type = CAT->getElementType();
    arraySizeMultiplier *= CAT->getSize();
  }

  CharUnits typeSize = CGF.getContext().getTypeSizeInChars(type);
  llvm::APInt typeSizeMultiplier(sizeWidth, typeSize.getQuantity());
  typeSizeMultiplier *= arraySizeMultiplier;

  // Figure out the cookie size.
  llvm::APInt cookieSize(sizeWidth,
                         CalculateCookiePadding(CGF, e).getQuantity());

  // This will be a size_t.
  mlir::Value size;

  // Emit the array size expression.
  // We multiply the size of all dimensions for NumElements.
  // e.g for 'int[2][3]', ElemType is 'int' and NumElements is 6.
  const Expr *arraySize = *e->getArraySize();
  mlir::Attribute constNumElements =
      ConstantEmitter(CGF.CGM, &CGF)
          .tryEmitAbstract(arraySize, arraySize->getType());
  if (constNumElements) {
    // Get an APInt from the constant
    const llvm::APInt &count =
        mlir::cast<cir::IntAttr>(constNumElements).getValue();

    unsigned numElementsWidth = count.getBitWidth();
    bool hasAnyOverflow = false;

    // The equivalent code in CodeGen/CGExprCXX.cpp handles these cases as
    // overflow, but they should never happen. The size argument is implicitly
    // cast to a size_t, so it can never be negative and numElementsWidth will
    // always equal sizeWidth.
    assert(!count.isNegative() && "Expected non-negative array size");
    assert(numElementsWidth == sizeWidth &&
           "Expected a size_t array size constant");

    // Okay, compute a count at the right width.
    llvm::APInt adjustedCount = count.zextOrTrunc(sizeWidth);

    // Scale numElements by that.  This might overflow, but we don't
    // care because it only overflows if allocationSize does, too, and
    // if that overflows then we shouldn't use this.
    // This emits a constant that may not be used, but we can't tell here
    // whether it will be needed or not.
    numElements =
        CGF.getBuilder().getConstInt(Loc, adjustedCount * arraySizeMultiplier);

    // Compute the size before cookie, and track whether it overflowed.
    bool overflow;
    llvm::APInt allocationSize =
        adjustedCount.umul_ov(typeSizeMultiplier, overflow);

    // Sema prevents us from hitting this case
    assert(!overflow && "Overflow in array allocation size");

    // Add in the cookie, and check whether it's overflowed.
    if (cookieSize != 0) {
      // Save the current size without a cookie.  This shouldn't be
      // used if there was overflow.
      sizeWithoutCookie = CGF.getBuilder().getConstInt(
          Loc, allocationSize.zextOrTrunc(sizeWidth));

      allocationSize = allocationSize.uadd_ov(cookieSize, overflow);
      hasAnyOverflow |= overflow;
    }

    // On overflow, produce a -1 so operator new will fail.
    if (hasAnyOverflow) {
      size =
          CGF.getBuilder().getConstInt(Loc, llvm::APInt::getAllOnes(sizeWidth));
    } else {
      size = CGF.getBuilder().getConstInt(Loc, allocationSize);
    }
  } else {
    // Create a value for the variable number of elements
    numElements = CGF.emitScalarExpr(*e->getArraySize());
    auto numElementsType = mlir::cast<cir::IntType>(numElements.getType());
    unsigned numElementsWidth = numElementsType.getWidth();

    // We might need check for overflow.

    mlir::Value hasOverflow = nullptr;
    // The clang LLVM IR codegen checks for the size variable being signed,
    // having a smaller width than size_t, and having a larger width than
    // size_t. However, the AST implicitly casts the size variable to size_t
    // so none of these conditions will ever be met.
    bool isSigned =
        (*e->getArraySize())->getType()->isSignedIntegerOrEnumerationType();
    assert(!isSigned && (numElementsWidth == sizeWidth) &&
           (numElements.getType() == CGF.SizeTy) &&
           "Expected array size to be implicitly cast to size_t!");

    // There are up to three conditions we need to test for:
    // 1) if minElements > 0, we need to check whether numElements is smaller
    //    than that.
    // 2) we need to compute
    //      sizeWithoutCookie := numElements * typeSizeMultiplier
    //    and check whether it overflows; and
    // 3) if we need a cookie, we need to compute
    //      size := sizeWithoutCookie + cookieSize
    //    and check whether it overflows.

    if (minElements) {
      // Don't allow allocation of fewer elements than we have initializers.
      if (!hasOverflow) {
        // FIXME: Avoid creating this twice. It may happen above.
        mlir::Value minElementsV = CGF.getBuilder().getConstInt(
            Loc, llvm::APInt(sizeWidth, minElements));
        hasOverflow = CGF.getBuilder().createCompare(Loc, cir::CmpOpKind::lt,
                                                     numElements, minElementsV);
      }
    }

    size = numElements;

    // Multiply by the type size if necessary.  This multiplier
    // includes all the factors for nested arrays.
    //
    // This step also causes numElements to be scaled up by the
    // nested-array factor if necessary.  Overflow on this computation
    // can be ignored because the result shouldn't be used if
    // allocation fails.
    if (typeSizeMultiplier != 1) {
      mlir::Value tsmV = CGF.getBuilder().getConstInt(Loc, typeSizeMultiplier);
      auto mulResult = CGF.getBuilder().createBinOpOverflowOp(
          Loc, mlir::cast<cir::IntType>(CGF.SizeTy),
          cir::BinOpOverflowKind::Mul, size, tsmV);

      if (hasOverflow)
        hasOverflow =
            CGF.getBuilder().createOr(hasOverflow, mulResult.overflow);
      else
        hasOverflow = mulResult.overflow;

      size = mulResult.result;

      // Also scale up numElements by the array size multiplier.
      if (arraySizeMultiplier != 1) {
        // If the base element type size is 1, then we can re-use the
        // multiply we just did.
        if (typeSize.isOne()) {
          assert(arraySizeMultiplier == typeSizeMultiplier);
          numElements = size;

          // Otherwise we need a separate multiply.
        } else {
          mlir::Value asmV =
              CGF.getBuilder().getConstInt(Loc, arraySizeMultiplier);
          numElements = CGF.getBuilder().createMul(numElements, asmV);
        }
      }
    } else {
      // numElements doesn't need to be scaled.
      assert(arraySizeMultiplier == 1);
    }

    // Add in the cookie size if necessary.
    if (cookieSize != 0) {
      sizeWithoutCookie = size;
      mlir::Value cookieSizeV = CGF.getBuilder().getConstInt(Loc, cookieSize);
      auto addResult = CGF.getBuilder().createBinOpOverflowOp(
          Loc, mlir::cast<cir::IntType>(CGF.SizeTy),
          cir::BinOpOverflowKind::Add, size, cookieSizeV);

      if (hasOverflow)
        hasOverflow =
            CGF.getBuilder().createOr(hasOverflow, addResult.overflow);
      else
        hasOverflow = addResult.overflow;

      size = addResult.result;
    }

    // If we had any possibility of dynamic overflow, make a select to
    // overwrite 'size' with an all-ones value, which should cause
    // operator new to throw.
    if (hasOverflow) {
      mlir::Value allOnes =
          CGF.getBuilder().getConstInt(Loc, llvm::APInt::getAllOnes(sizeWidth));
      size = CGF.getBuilder().createSelect(Loc, hasOverflow, allOnes, size);
    }
  }

  if (cookieSize == 0)
    sizeWithoutCookie = size;
  else
    assert(sizeWithoutCookie && "didn't set sizeWithoutCookie?");

  return size;
}

namespace {
/// A cleanup to call the given 'operator delete' function upon abnormal
/// exit from a new expression. Templated on a traits type that deals with
/// ensuring that the arguments dominate the cleanup if necessary.
template <typename Traits>
class CallDeleteDuringNew final : public EHScopeStack::Cleanup {
  /// Type used to hold llvm::Value*s.
  typedef typename Traits::ValueTy ValueTy;
  /// Type used to hold RValues.
  typedef typename Traits::RValueTy RValueTy;
  struct PlacementArg {
    RValueTy ArgValue;
    QualType ArgType;
  };

  unsigned NumPlacementArgs : 31;
  unsigned PassAlignmentToPlacementDelete : 1;
  const FunctionDecl *OperatorDelete;
  ValueTy Ptr;
  ValueTy AllocSize;
  CharUnits AllocAlign;

  PlacementArg *getPlacementArgs() {
    return reinterpret_cast<PlacementArg *>(this + 1);
  }

public:
  static size_t getExtraSize(size_t NumPlacementArgs) {
    return NumPlacementArgs * sizeof(PlacementArg);
  }

  CallDeleteDuringNew(size_t NumPlacementArgs,
                      const FunctionDecl *OperatorDelete, ValueTy Ptr,
                      ValueTy AllocSize, bool PassAlignmentToPlacementDelete,
                      CharUnits AllocAlign)
      : NumPlacementArgs(NumPlacementArgs),
        PassAlignmentToPlacementDelete(PassAlignmentToPlacementDelete),
        OperatorDelete(OperatorDelete), Ptr(Ptr), AllocSize(AllocSize),
        AllocAlign(AllocAlign) {}

  void setPlacementArg(unsigned I, RValueTy Arg, QualType Type) {
    assert(I < NumPlacementArgs && "index out of range");
    getPlacementArgs()[I] = {Arg, Type};
  }

  void Emit(CIRGenFunction &CGF, Flags flags) override {
    const auto *FPT = OperatorDelete->getType()->castAs<FunctionProtoType>();
    CallArgList DeleteArgs;

    // The first argument is always a void* (or C* for a destroying operator
    // delete for class type C).
    DeleteArgs.add(Traits::get(CGF, Ptr), FPT->getParamType(0));

    // Figure out what other parameters we should be implicitly passing.
    UsualDeleteParams Params;
    if (NumPlacementArgs) {
      // A placement deallocation function is implicitly passed an alignment
      // if the placement allocation function was, but is never passed a size.
      Params.Alignment = PassAlignmentToPlacementDelete;
    } else {
      // For a non-placement new-expression, 'operator delete' can take a
      // size and/or an alignment if it has the right parameters.
      Params = getUsualDeleteParams(OperatorDelete);
    }

    assert(!Params.DestroyingDelete &&
           "should not call destroying delete in a new-expression");

    // The second argument can be a std::size_t (for non-placement delete).
    if (Params.Size)
      DeleteArgs.add(Traits::get(CGF, AllocSize),
                     CGF.getContext().getSizeType());

    // The next (second or third) argument can be a std::align_val_t, which
    // is an enum whose underlying type is std::size_t.
    // FIXME: Use the right type as the parameter type. Note that in a call
    // to operator delete(size_t, ...), we may not have it available.
    if (Params.Alignment) {
      llvm_unreachable("NYI");
    }

    // Pass the rest of the arguments, which must match exactly.
    for (unsigned I = 0; I != NumPlacementArgs; ++I) {
      auto Arg = getPlacementArgs()[I];
      DeleteArgs.add(Traits::get(CGF, Arg.ArgValue), Arg.ArgType);
    }

    // Call 'operator delete'.
    emitNewDeleteCall(CGF, OperatorDelete, FPT, DeleteArgs);
  }
};
} // namespace

/// Enter a cleanup to call 'operator delete' if the initializer in a
/// new-expression throws.
static void EnterNewDeleteCleanup(CIRGenFunction &CGF, const CXXNewExpr *E,
                                  Address NewPtr, mlir::Value AllocSize,
                                  CharUnits AllocAlign,
                                  const CallArgList &NewArgs) {
  unsigned NumNonPlacementArgs = E->passAlignment() ? 2 : 1;

  // If we're not inside a conditional branch, then the cleanup will
  // dominate and we can do the easier (and more efficient) thing.
  if (!CGF.isInConditionalBranch()) {
    struct DirectCleanupTraits {
      typedef mlir::Value ValueTy;
      typedef RValue RValueTy;
      static RValue get(CIRGenFunction &, ValueTy V) { return RValue::get(V); }
      static RValue get(CIRGenFunction &, RValueTy V) { return V; }
    };

    typedef CallDeleteDuringNew<DirectCleanupTraits> DirectCleanup;

    DirectCleanup *Cleanup = CGF.EHStack.pushCleanupWithExtra<DirectCleanup>(
        EHCleanup, E->getNumPlacementArgs(), E->getOperatorDelete(),
        NewPtr.getPointer(), AllocSize, E->passAlignment(), AllocAlign);
    for (unsigned I = 0, N = E->getNumPlacementArgs(); I != N; ++I) {
      auto &Arg = NewArgs[I + NumNonPlacementArgs];
      Cleanup->setPlacementArg(
          I, Arg.getRValue(CGF, CGF.getLoc(E->getSourceRange())), Arg.Ty);
    }

    return;
  }

  // Otherwise, we need to save all this stuff.
  DominatingValue<RValue>::saved_type SavedNewPtr =
      DominatingValue<RValue>::save(CGF, RValue::get(NewPtr.getPointer()));
  DominatingValue<RValue>::saved_type SavedAllocSize =
      DominatingValue<RValue>::save(CGF, RValue::get(AllocSize));

  struct ConditionalCleanupTraits {
    typedef DominatingValue<RValue>::saved_type ValueTy;
    typedef DominatingValue<RValue>::saved_type RValueTy;
    static RValue get(CIRGenFunction &CGF, ValueTy V) { return V.restore(CGF); }
  };
  typedef CallDeleteDuringNew<ConditionalCleanupTraits> ConditionalCleanup;

  ConditionalCleanup *Cleanup =
      CGF.EHStack.pushCleanupWithExtra<ConditionalCleanup>(
          EHCleanup, E->getNumPlacementArgs(), E->getOperatorDelete(),
          SavedNewPtr, SavedAllocSize, E->passAlignment(), AllocAlign);
  for (unsigned I = 0, N = E->getNumPlacementArgs(); I != N; ++I) {
    auto &Arg = NewArgs[I + NumNonPlacementArgs];
    Cleanup->setPlacementArg(
        I,
        DominatingValue<RValue>::save(
            CGF, Arg.getRValue(CGF, CGF.getLoc(E->getSourceRange()))),
        Arg.Ty);
  }

  CGF.initFullExprCleanup();
}

static void StoreAnyExprIntoOneUnit(CIRGenFunction &CGF, const Expr *Init,
                                    QualType AllocType, Address NewPtr,
                                    AggValueSlot::Overlap_t MayOverlap) {
  // FIXME: Refactor with emitExprAsInit.
  switch (CGF.getEvaluationKind(AllocType)) {
  case cir::TEK_Scalar:
    CGF.emitScalarInit(Init, CGF.getLoc(Init->getSourceRange()),
                       CGF.makeAddrLValue(NewPtr, AllocType), false);
    return;
  case cir::TEK_Complex:
    llvm_unreachable("NYI");
    return;
  case cir::TEK_Aggregate: {
    AggValueSlot Slot = AggValueSlot::forAddr(
        NewPtr, AllocType.getQualifiers(), AggValueSlot::IsDestructed,
        AggValueSlot::DoesNotNeedGCBarriers, AggValueSlot::IsNotAliased,
        MayOverlap, AggValueSlot::IsNotZeroed,
        AggValueSlot::IsSanitizerChecked);
    CGF.emitAggExpr(Init, Slot);
    return;
  }
  }
  llvm_unreachable("bad evaluation kind");
}

void CIRGenFunction::emitNewArrayInitializer(
    const CXXNewExpr *E, QualType ElementType, mlir::Type ElementTy,
    Address BeginPtr, mlir::Value NumElements,
    mlir::Value AllocSizeWithoutCookie) {
  // If we have a type with trivial initialization and no initializer,
  // there's nothing to do.
  if (!E->hasInitializer())
    return;

  Address CurPtr = BeginPtr;

  unsigned InitListElements = 0;

  const Expr *Init = E->getInitializer();
  Address EndOfInit = Address::invalid();
  QualType::DestructionKind DtorKind = ElementType.isDestructedType();
  CleanupDeactivationScope deactivation(*this);

  CharUnits ElementSize = getContext().getTypeSizeInChars(ElementType);

  // Attempt to perform zero-initialization using memset.
  auto TryMemsetInitialization = [&]() -> bool {
    auto Loc = NumElements.getLoc();

    // FIXME: If the type is a pointer-to-data-member under the Itanium ABI,
    // we can initialize with a memset to -1.
    if (!CGM.getTypes().isZeroInitializable(ElementType))
      return false;

    // Optimization: since zero initialization will just set the memory
    // to all zeroes, generate a single memset to do it in one shot.

    // Subtract out the size of any elements we've already initialized.
    auto RemainingSize = AllocSizeWithoutCookie;
    if (InitListElements) {
      // We know this can't overflow; we check this when doing the allocation.
      unsigned InitializedSize =
          getContext().getTypeSizeInChars(ElementType).getQuantity() *
          InitListElements;
      auto InitSizeOp =
          builder.getConstInt(Loc, RemainingSize.getType(), InitializedSize);
      RemainingSize = builder.createSub(RemainingSize, InitSizeOp);
    }

    // Create the memset.
    auto CastOp =
        builder.createPtrBitcast(CurPtr.getPointer(), builder.getVoidTy());
    builder.createMemSet(Loc, CastOp, builder.getUInt8(0, Loc), RemainingSize);
    return true;
  };

  const InitListExpr *ILE = dyn_cast<InitListExpr>(Init);
  const CXXParenListInitExpr *CPLIE = nullptr;
  const StringLiteral *SL = nullptr;
  const ObjCEncodeExpr *OCEE = nullptr;
  const Expr *IgnoreParen = nullptr;
  if (!ILE) {
    IgnoreParen = Init->IgnoreParenImpCasts();
    CPLIE = dyn_cast<CXXParenListInitExpr>(IgnoreParen);
    SL = dyn_cast<StringLiteral>(IgnoreParen);
    OCEE = dyn_cast<ObjCEncodeExpr>(IgnoreParen);
  }

  // If the initializer is an initializer list, first do the explicit elements.
  if (ILE || CPLIE || SL || OCEE) {
    // Initializing from a (braced) string literal is a special case; the init
    // list element does not initialize a (single) array element.
    if ((ILE && ILE->isStringLiteralInit()) || SL || OCEE) {
      llvm_unreachable("NYI");
    }

    ArrayRef<const Expr *> InitExprs =
        ILE ? ILE->inits() : CPLIE->getInitExprs();
    InitListElements = InitExprs.size();

    // If this is a multi-dimensional array new, we will initialize multiple
    // elements with each init list element.
    QualType AllocType = E->getAllocatedType();
    if (const ConstantArrayType *CAT = dyn_cast_or_null<ConstantArrayType>(
            AllocType->getAsArrayTypeUnsafe())) {
      llvm_unreachable("NYI");
    }

    // Enter a partial-destruction Cleanup if necessary.
    if (DtorKind) {
      llvm_unreachable("NYI");
    }

    CharUnits StartAlign = CurPtr.getAlignment();
    unsigned i = 0;
    for (const Expr *IE : InitExprs) {
      if (EndOfInit.isValid()) {
        // This will involve DTor handling.
        llvm_unreachable("NYI");
      }
      // FIXME: If the last initializer is an incomplete initializer list for
      // an array, and we have an array filler, we can fold together the two
      // initialization loops.
      StoreAnyExprIntoOneUnit(*this, IE, IE->getType(), CurPtr,
                              AggValueSlot::DoesNotOverlap);
      auto Loc = getLoc(IE->getExprLoc());
      auto CastOp = builder.createPtrBitcast(CurPtr.getPointer(),
                                             convertTypeForMem(AllocType));
      auto OffsetOp = builder.getSignedInt(Loc, 1, /*width=*/32);
      auto DataPtr = builder.createPtrStride(Loc, CastOp, OffsetOp);
      CurPtr = Address(DataPtr, CurPtr.getType(),
                       StartAlign.alignmentAtOffset((++i) * ElementSize));
    }

    // The remaining elements are filled with the array filler expression.
    Init = ILE ? ILE->getArrayFiller() : CPLIE->getArrayFiller();

    // Extract the initializer for the individual array elements by pulling
    // out the array filler from all the nested initializer lists. This avoids
    // generating a nested loop for the initialization.
    while (Init && Init->getType()->isConstantArrayType()) {
      auto *SubILE = dyn_cast<InitListExpr>(Init);
      if (!SubILE)
        break;
      assert(SubILE->getNumInits() == 0 && "explicit inits in array filler?");
      Init = SubILE->getArrayFiller();
    }

    // Switch back to initializing one base element at a time.
    CurPtr = CurPtr.withElementType(BeginPtr.getElementType());
  }

  // If all elements have already been initialized, skip any further
  // initialization.
  auto ConstOp = dyn_cast<cir::ConstantOp>(NumElements.getDefiningOp());
  if (ConstOp) {
    auto ConstIntAttr = mlir::dyn_cast<cir::IntAttr>(ConstOp.getValue());
    // Just skip out if the constant count is zero.
    if (ConstIntAttr && ConstIntAttr.getUInt() <= InitListElements)
      return;
  }

  assert(Init && "have trailing elements to initialize but no initializer");

  // If this is a constructor call, try to optimize it out, and failing that
  // emit a single loop to initialize all remaining elements.
  if (const CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(Init)) {
    CXXConstructorDecl *Ctor = CCE->getConstructor();
    if (Ctor->isTrivial()) {
      // If new expression did not specify value-initialization, then there
      // is no initialization.
      if (!CCE->requiresZeroInitialization() || Ctor->getParent()->isEmpty())
        return;

      llvm_unreachable("NYI");
    }

    llvm_unreachable("NYI");
  }

  // If this is value-initialization, we can usually use memset.
  if (isa<ImplicitValueInitExpr>(Init)) {
    if (TryMemsetInitialization())
      return;
    llvm_unreachable("NYI");
  }
  llvm_unreachable("NYI");
}

static void emitNewInitializer(CIRGenFunction &CGF, const CXXNewExpr *E,
                               QualType ElementType, mlir::Type ElementTy,
                               Address NewPtr, mlir::Value NumElements,
                               mlir::Value AllocSizeWithoutCookie) {
  assert(!cir::MissingFeatures::generateDebugInfo());
  if (E->isArray()) {
    CGF.emitNewArrayInitializer(E, ElementType, ElementTy, NewPtr, NumElements,
                                AllocSizeWithoutCookie);
  } else if (const Expr *Init = E->getInitializer()) {
    StoreAnyExprIntoOneUnit(CGF, Init, E->getAllocatedType(), NewPtr,
                            AggValueSlot::DoesNotOverlap);
  }
}

namespace {
/// Calls the given 'operator delete' on a single object.
struct CallObjectDelete final : EHScopeStack::Cleanup {
  mlir::Value Ptr;
  const FunctionDecl *OperatorDelete;
  QualType ElementType;

  CallObjectDelete(mlir::Value Ptr, const FunctionDecl *OperatorDelete,
                   QualType ElementType)
      : Ptr(Ptr), OperatorDelete(OperatorDelete), ElementType(ElementType) {}

  void Emit(CIRGenFunction &CGF, Flags flags) override {
    CGF.emitDeleteCall(OperatorDelete, Ptr, ElementType);
  }
};
} // namespace

/// Emit the code for deleting a single object.
/// \return \c true if we started emitting UnconditionalDeleteBlock, \c false
/// if not.
static bool EmitObjectDelete(CIRGenFunction &CGF, const CXXDeleteExpr *DE,
                             Address Ptr, QualType ElementType) {
  // C++11 [expr.delete]p3:
  //   If the static type of the object to be deleted is different from its
  //   dynamic type, the static type shall be a base class of the dynamic type
  //   of the object to be deleted and the static type shall have a virtual
  //   destructor or the behavior is undefined.
  CGF.emitTypeCheck(CIRGenFunction::TCK_MemberCall, DE->getExprLoc(),
                    Ptr.getPointer(), ElementType);

  const FunctionDecl *OperatorDelete = DE->getOperatorDelete();
  assert(!OperatorDelete->isDestroyingOperatorDelete());

  // Find the destructor for the type, if applicable.  If the
  // destructor is virtual, we'll just emit the vcall and return.
  const CXXDestructorDecl *Dtor = nullptr;
  if (const RecordType *RT = ElementType->getAs<RecordType>()) {
    CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
    if (RD->hasDefinition() && !RD->hasTrivialDestructor()) {
      Dtor = RD->getDestructor();

      if (Dtor->isVirtual()) {
        bool UseVirtualCall = true;
        const Expr *Base = DE->getArgument();
        if (auto *DevirtualizedDtor = dyn_cast_or_null<const CXXDestructorDecl>(
                Dtor->getDevirtualizedMethod(
                    Base, CGF.CGM.getLangOpts().AppleKext))) {
          UseVirtualCall = false;
          const CXXRecordDecl *DevirtualizedClass =
              DevirtualizedDtor->getParent();
          if (declaresSameEntity(getCXXRecord(Base), DevirtualizedClass)) {
            // Devirtualized to the class of the base type (the type of the
            // whole expression).
            Dtor = DevirtualizedDtor;
          } else {
            // Devirtualized to some other type. Would need to cast the this
            // pointer to that type but we don't have support for that yet, so
            // do a virtual call. FIXME: handle the case where it is
            // devirtualized to the derived type (the type of the inner
            // expression) as in EmitCXXMemberOrOperatorMemberCallExpr.
            UseVirtualCall = true;
          }
        }
        if (UseVirtualCall) {
          llvm_unreachable("NYI");
          return false;
        }
      }
    }
  }

  // Make sure that we call delete even if the dtor throws.
  // This doesn't have to a conditional cleanup because we're going
  // to pop it off in a second.
  CGF.EHStack.pushCleanup<CallObjectDelete>(
      NormalAndEHCleanup, Ptr.getPointer(), OperatorDelete, ElementType);

  if (Dtor) {
    llvm_unreachable("NYI");
  } else if (auto Lifetime = ElementType.getObjCLifetime()) {
    switch (Lifetime) {
    case Qualifiers::OCL_None:
    case Qualifiers::OCL_ExplicitNone:
    case Qualifiers::OCL_Autoreleasing:
      break;

    case Qualifiers::OCL_Strong:
      llvm_unreachable("NYI");
      break;

    case Qualifiers::OCL_Weak:
      llvm_unreachable("NYI");
      break;
    }
  }

  // In traditional LLVM codegen null checks are emitted to save a delete call.
  // In CIR we optimize for size by default, the null check should be added into
  // this function callers.
  assert(!cir::MissingFeatures::emitNullCheckForDeleteCalls());

  CGF.PopCleanupBlock();
  return false;
}

void CIRGenFunction::emitCXXDeleteExpr(const CXXDeleteExpr *E) {
  const Expr *Arg = E->getArgument();
  Address Ptr = emitPointerWithAlignment(Arg);

  // Null check the pointer.
  //
  // We could avoid this null check if we can determine that the object
  // destruction is trivial and doesn't require an array cookie; we can
  // unconditionally perform the operator delete call in that case. For now, we
  // assume that deleted pointers are null rarely enough that it's better to
  // keep the branch. This might be worth revisiting for a -O0 code size win.
  //
  // CIR note: emit the code size friendly by default for now, such as mentioned
  // in `EmitObjectDelete`.
  assert(!cir::MissingFeatures::emitNullCheckForDeleteCalls());
  QualType DeleteTy = E->getDestroyedType();

  // A destroying operator delete overrides the entire operation of the
  // delete expression.
  if (E->getOperatorDelete()->isDestroyingOperatorDelete()) {
    llvm_unreachable("NYI");
    return;
  }

  // In CodeGen:
  // We might be deleting a pointer to array.  If so, GEP down to the
  // first non-array element.
  // (this assumes that A(*)[3][7] is converted to [3 x [7 x %A]]*)
  // In CIRGen: we handle this differently because the deallocation of
  // array highly relates to the array cookies, which is ABI sensitive,
  // we plan to handle it in LoweringPreparePass and the corresponding
  // ABI part.
  if (DeleteTy->isConstantArrayType()) {
    // Nothing to do here, keep it for skeleton comparison sake.
  }

  assert(convertTypeForMem(DeleteTy) == Ptr.getElementType());

  if (E->isArrayForm()) {
    builder.create<cir::DeleteArrayOp>(Ptr.getPointer().getLoc(),
                                       Ptr.getPointer());
  } else {
    (void)EmitObjectDelete(*this, E, Ptr, DeleteTy);
  }
}

mlir::Value CIRGenFunction::emitCXXNewExpr(const CXXNewExpr *E) {
  // The element type being allocated.
  QualType allocType = getContext().getBaseElementType(E->getAllocatedType());

  // 1. Build a call to the allocation function.
  FunctionDecl *allocator = E->getOperatorNew();

  // If there is a brace-initializer, cannot allocate fewer elements than inits.
  unsigned minElements = 0;
  if (E->isArray() && E->hasInitializer()) {
    const InitListExpr *ILE = dyn_cast<InitListExpr>(E->getInitializer());
    if (ILE && ILE->isStringLiteralInit())
      minElements =
          cast<ConstantArrayType>(ILE->getType()->getAsArrayTypeUnsafe())
              ->getSize()
              .getZExtValue();
    else if (ILE)
      minElements = ILE->getNumInits();
  }

  mlir::Value numElements = nullptr;
  mlir::Value allocSizeWithoutCookie = nullptr;
  mlir::Value allocSize = emitCXXNewAllocSize(
      *this, E, minElements, numElements, allocSizeWithoutCookie);
  CharUnits allocAlign = getContext().getTypeAlignInChars(allocType);

  // Emit the allocation call.
  Address allocation = Address::invalid();
  CallArgList allocatorArgs;
  if (allocator->isReservedGlobalPlacementOperator()) {
    // If the allocator is a global placement operator, just
    // "inline" it directly.
    assert(E->getNumPlacementArgs() == 1);
    const Expr *arg = *E->placement_arguments().begin();

    LValueBaseInfo BaseInfo;
    allocation = emitPointerWithAlignment(arg, &BaseInfo);

    // The pointer expression will, in many cases, be an opaque void*.
    // In these cases, discard the computed alignment and use the
    // formal alignment of the allocated type.
    if (BaseInfo.getAlignmentSource() != AlignmentSource::Decl)
      allocation = allocation.withAlignment(allocAlign);

    // Set up allocatorArgs for the call to operator delete if it's not
    // the reserved global operator.
    if (E->getOperatorDelete() &&
        !E->getOperatorDelete()->isReservedGlobalPlacementOperator()) {
      allocatorArgs.add(RValue::get(allocSize), getContext().getSizeType());
      allocatorArgs.add(RValue::get(allocation.getPointer()), arg->getType());
    }
  } else {
    const FunctionProtoType *allocatorType =
        allocator->getType()->castAs<FunctionProtoType>();
    unsigned ParamsToSkip = 0;

    // The allocation size is the first argument.
    QualType sizeType = getContext().getSizeType();
    allocatorArgs.add(RValue::get(allocSize), sizeType);
    ++ParamsToSkip;

    if (allocSize != allocSizeWithoutCookie) {
      CharUnits cookieAlign = getSizeAlign(); // FIXME: Ask the ABI.
      allocAlign = std::max(allocAlign, cookieAlign);
    }

    // The allocation alignment may be passed as the second argument.
    if (E->passAlignment()) {
      llvm_unreachable("NYI");
    }

    // FIXME: Why do we not pass a CalleeDecl here?
    emitCallArgs(allocatorArgs, allocatorType, E->placement_arguments(),
                 /*AC*/
                 AbstractCallee(),
                 /*ParamsToSkip*/
                 ParamsToSkip);
    RValue RV =
        emitNewDeleteCall(*this, allocator, allocatorType, allocatorArgs);

    // Set !heapallocsite metadata on the call to operator new.
    assert(!cir::MissingFeatures::generateDebugInfo());

    // If this was a call to a global replaceable allocation function that does
    // not take an alignment argument, the allocator is known to produce storage
    // that's suitably aligned for any object that fits, up to a known
    // threshold. Otherwise assume it's suitably aligned for the allocated type.
    CharUnits allocationAlign = allocAlign;
    if (!E->passAlignment() &&
        allocator->isReplaceableGlobalAllocationFunction()) {
      auto &Target = CGM.getASTContext().getTargetInfo();
      unsigned AllocatorAlign = llvm::bit_floor(std::min<uint64_t>(
          Target.getNewAlign(), getContext().getTypeSize(allocType)));
      allocationAlign = std::max(
          allocationAlign, getContext().toCharUnitsFromBits(AllocatorAlign));
    }

    allocation = Address(RV.getScalarVal(), UInt8Ty, allocationAlign);
  }

  // Emit a null check on the allocation result if the allocation
  // function is allowed to return null (because it has a non-throwing
  // exception spec or is the reserved placement new) and we have an
  // interesting initializer will be running sanitizers on the initialization.
  bool nullCheck = E->shouldNullCheckAllocation() &&
                   (!allocType.isPODType(getContext()) || E->hasInitializer() ||
                    sanitizePerformTypeCheck());

  // The null-check means that the initializer is conditionally
  // evaluated.
  mlir::OpBuilder::InsertPoint ifBody, postIfBody, preIfBody;
  mlir::Value nullCmpResult;
  mlir::Location loc = getLoc(E->getSourceRange());

  if (nullCheck) {
    mlir::Value nullPtr =
        builder.getNullPtr(allocation.getPointer().getType(), loc);
    nullCmpResult = builder.createCompare(loc, cir::CmpOpKind::ne,
                                          allocation.getPointer(), nullPtr);
    preIfBody = builder.saveInsertionPoint();
    builder.create<cir::IfOp>(loc, nullCmpResult,
                              /*withElseRegion=*/false,
                              [&](mlir::OpBuilder &, mlir::Location) {
                                ifBody = builder.saveInsertionPoint();
                              });
    postIfBody = builder.saveInsertionPoint();
  }

  // Make sure the conditional evaluation uses the insertion
  // point right before the if check.
  mlir::OpBuilder::InsertPoint ip = builder.saveInsertionPoint();
  if (ifBody.isSet()) {
    builder.setInsertionPointAfterValue(nullCmpResult);
    ip = builder.saveInsertionPoint();
  }
  ConditionalEvaluation conditional(ip);

  // All the actual work to be done should be placed inside the IfOp above,
  // so change the insertion point over there.
  if (ifBody.isSet()) {
    conditional.begin(*this);
    builder.restoreInsertionPoint(ifBody);
  }

  // If there's an operator delete, enter a cleanup to call it if an
  // exception is thrown.
  EHScopeStack::stable_iterator operatorDeleteCleanup;
  [[maybe_unused]] mlir::Operation *cleanupDominator = nullptr;
  if (E->getOperatorDelete() &&
      !E->getOperatorDelete()->isReservedGlobalPlacementOperator()) {
    EnterNewDeleteCleanup(*this, E, allocation, allocSize, allocAlign,
                          allocatorArgs);
    operatorDeleteCleanup = EHStack.stable_begin();
    cleanupDominator =
        builder.create<cir::UnreachableOp>(getLoc(E->getSourceRange()))
            .getOperation();
  }

  assert((allocSize == allocSizeWithoutCookie) ==
         CalculateCookiePadding(*this, E).isZero());
  if (allocSize != allocSizeWithoutCookie) {
    assert(E->isArray());
    allocation = CGM.getCXXABI().initializeArrayCookie(
        *this, allocation, numElements, E, allocType);
  }

  mlir::Type elementTy;
  Address result = Address::invalid();
  auto createCast = [&]() {
    elementTy = convertTypeForMem(allocType);
    result = builder.createElementBitCast(getLoc(E->getSourceRange()),
                                          allocation, elementTy);
  };

  if (preIfBody.isSet()) {
    // Generate any cast before the if condition check on the null because the
    // result can be used after the if body and should dominate all potential
    // uses.
    mlir::OpBuilder::InsertionGuard guard(builder);
    assert(nullCmpResult && "expected");
    builder.setInsertionPointAfterValue(nullCmpResult);
    createCast();
  } else {
    createCast();
  }

  // Passing pointer through launder.invariant.group to avoid propagation of
  // vptrs information which may be included in previous type.
  // To not break LTO with different optimizations levels, we do it regardless
  // of optimization level.
  if (CGM.getCodeGenOpts().StrictVTablePointers &&
      allocator->isReservedGlobalPlacementOperator())
    llvm_unreachable("NYI");

  // Emit sanitizer checks for pointer value now, so that in the case of an
  // array it was checked only once and not at each constructor call. We may
  // have already checked that the pointer is non-null.
  // FIXME: If we have an array cookie and a potentially-throwing allocator,
  // we'll null check the wrong pointer here.
  SanitizerSet SkippedChecks;
  SkippedChecks.set(SanitizerKind::Null, nullCheck);
  emitTypeCheck(CIRGenFunction::TCK_ConstructorCall,
                E->getAllocatedTypeSourceInfo()->getTypeLoc().getBeginLoc(),
                result.getPointer(), allocType, result.getAlignment(),
                SkippedChecks, numElements);

  emitNewInitializer(*this, E, allocType, elementTy, result, numElements,
                     allocSizeWithoutCookie);
  auto resultPtr = result.getPointer();

  // Deactivate the 'operator delete' cleanup if we finished
  // initialization.
  if (operatorDeleteCleanup.isValid()) {
    // FIXME: enable cleanupDominator above before implementing this.
    DeactivateCleanupBlock(operatorDeleteCleanup, cleanupDominator);
    if (cleanupDominator)
      cleanupDominator->erase();
  }

  if (nullCheck) {
    conditional.end(*this);
    // resultPtr is already updated in the first null check phase.

    // Reset insertion point to resume back to post ifOp.
    if (postIfBody.isSet()) {
      builder.create<cir::YieldOp>(loc);
      builder.restoreInsertionPoint(postIfBody);
    }
  }

  return resultPtr;
}

RValue CIRGenFunction::emitCXXDestructorCall(GlobalDecl Dtor,
                                             const CIRGenCallee &Callee,
                                             mlir::Value This, QualType ThisTy,
                                             mlir::Value ImplicitParam,
                                             QualType ImplicitParamTy,
                                             const CallExpr *CE) {
  const CXXMethodDecl *DtorDecl = cast<CXXMethodDecl>(Dtor.getDecl());

  assert(!ThisTy.isNull());
  assert(ThisTy->getAsCXXRecordDecl() == DtorDecl->getParent() &&
         "Pointer/Object mixup");

  LangAS SrcAS = ThisTy.getAddressSpace();
  LangAS DstAS = DtorDecl->getMethodQualifiers().getAddressSpace();
  if (SrcAS != DstAS) {
    llvm_unreachable("NYI");
  }

  CallArgList Args;
  commonBuildCXXMemberOrOperatorCall(*this, DtorDecl, This, ImplicitParam,
                                     ImplicitParamTy, CE, Args, nullptr);
  assert((CE || Dtor.getDecl()) && "expected source location provider");
  return emitCall(CGM.getTypes().arrangeCXXStructorDeclaration(Dtor), Callee,
                  ReturnValueSlot(), Args, nullptr, CE && CE == MustTailCall,
                  CE ? getLoc(CE->getExprLoc())
                     : getLoc(Dtor.getDecl()->getSourceRange()));
}

/// Emit a call to an operator new or operator delete function, as implicitly
/// created by new-expressions and delete-expressions.
static RValue emitNewDeleteCall(CIRGenFunction &CGF,
                                const FunctionDecl *CalleeDecl,
                                const FunctionProtoType *CalleeType,
                                const CallArgList &Args) {
  cir::CIRCallOpInterface CallOrTryCall;
  auto CalleePtr = CGF.CGM.GetAddrOfFunction(CalleeDecl);
  CIRGenCallee Callee =
      CIRGenCallee::forDirect(CalleePtr, GlobalDecl(CalleeDecl));
  RValue RV = CGF.emitCall(CGF.CGM.getTypes().arrangeFreeFunctionCall(
                               Args, CalleeType, /*ChainCall=*/false),
                           Callee, ReturnValueSlot(), Args, &CallOrTryCall);

  /// C++1y [expr.new]p10:
  ///   [In a new-expression,] an implementation is allowed to omit a call
  ///   to a replaceable global allocation function.
  ///
  /// We model such elidable calls with the 'builtin' attribute.
  assert(!cir::MissingFeatures::attributeBuiltin());
  return RV;
}

RValue CIRGenFunction::emitBuiltinNewDeleteCall(const FunctionProtoType *type,
                                                const CallExpr *theCall,
                                                bool isDelete) {
  CallArgList args;
  emitCallArgs(args, type, theCall->arguments());
  // Find the allocation or deallocation function that we're calling.
  ASTContext &astContext = getContext();
  DeclarationName name = astContext.DeclarationNames.getCXXOperatorName(
      isDelete ? OO_Delete : OO_New);

  for (auto *decl : astContext.getTranslationUnitDecl()->lookup(name))
    if (auto *fd = dyn_cast<FunctionDecl>(decl))
      if (astContext.hasSameType(fd->getType(), QualType(type, 0)))
        return emitNewDeleteCall(*this, fd, type, args);
  llvm_unreachable("predeclared global operator new/delete is missing");
}

void CIRGenFunction::emitDeleteCall(const FunctionDecl *DeleteFD,
                                    mlir::Value Ptr, QualType DeleteTy,
                                    mlir::Value NumElements,
                                    CharUnits CookieSize) {
  assert((!NumElements && CookieSize.isZero()) ||
         DeleteFD->getOverloadedOperator() == OO_Array_Delete);

  const auto *DeleteFTy = DeleteFD->getType()->castAs<FunctionProtoType>();
  CallArgList DeleteArgs;

  auto Params = getUsualDeleteParams(DeleteFD);
  auto ParamTypeIt = DeleteFTy->param_type_begin();

  // Pass the pointer itself.
  QualType ArgTy = *ParamTypeIt++;
  mlir::Value DeletePtr =
      builder.createBitcast(Ptr.getLoc(), Ptr, convertType(ArgTy));
  DeleteArgs.add(RValue::get(DeletePtr), ArgTy);

  // Pass the std::destroying_delete tag if present.
  mlir::Value DestroyingDeleteTag{};
  if (Params.DestroyingDelete) {
    llvm_unreachable("NYI");
  }

  // Pass the size if the delete function has a size_t parameter.
  if (Params.Size) {
    QualType SizeType = *ParamTypeIt++;
    CharUnits DeleteTypeSize = getContext().getTypeSizeInChars(DeleteTy);
    assert(SizeTy && "expected cir::IntType");
    auto Size = builder.getConstInt(*currSrcLoc, convertType(SizeType),
                                    DeleteTypeSize.getQuantity());

    // For array new, multiply by the number of elements.
    if (NumElements) {
      // Uncomment upon adding testcase.
      // Size = builder.createMul(Size, NumElements);
      llvm_unreachable("NYI");
    }

    // If there is a cookie, add the cookie size.
    if (!CookieSize.isZero()) {
      // Uncomment upon adding testcase.
      // builder.createBinop(
      //     Size, cir::BinOpKind::Add,
      //     builder.getConstInt(*currSrcLoc, SizeTy,
      //     CookieSize.getQuantity()));
      llvm_unreachable("NYI");
    }

    DeleteArgs.add(RValue::get(Size), SizeType);
  }

  // Pass the alignment if the delete function has an align_val_t parameter.
  if (Params.Alignment) {
    llvm_unreachable("NYI");
  }

  assert(ParamTypeIt == DeleteFTy->param_type_end() &&
         "unknown parameter to usual delete function");

  // Emit the call to delete.
  emitNewDeleteCall(*this, DeleteFD, DeleteFTy, DeleteArgs);

  // If call argument lowering didn't use the destroying_delete_t alloca,
  // remove it again.
  if (DestroyingDeleteTag && DestroyingDeleteTag.use_empty()) {
    llvm_unreachable("NYI"); // DestroyingDeleteTag->eraseFromParent();
  }
}

static mlir::Value emitDynamicCastToNull(CIRGenFunction &CGF,
                                         mlir::Location Loc, QualType DestTy) {
  mlir::Type DestCIRTy = CGF.convertType(DestTy);
  assert(mlir::isa<cir::PointerType>(DestCIRTy) &&
         "result of dynamic_cast should be a ptr");

  mlir::Value NullPtrValue = CGF.getBuilder().getNullPtr(DestCIRTy, Loc);

  if (!DestTy->isPointerType()) {
    auto *CurrentRegion = CGF.getBuilder().getBlock()->getParent();
    /// C++ [expr.dynamic.cast]p9:
    ///   A failed cast to reference type throws std::bad_cast
    CGF.CGM.getCXXABI().emitBadCastCall(CGF, Loc);

    // The call to bad_cast will terminate the current block. Create a new block
    // to hold any follow up code.
    CGF.getBuilder().createBlock(CurrentRegion, CurrentRegion->end());
  }

  return NullPtrValue;
}

mlir::Value CIRGenFunction::emitDynamicCast(Address ThisAddr,
                                            const CXXDynamicCastExpr *DCE) {
  auto loc = getLoc(DCE->getSourceRange());

  CGM.emitExplicitCastExprType(DCE, this);
  QualType destTy = DCE->getTypeAsWritten();
  QualType srcTy = DCE->getSubExpr()->getType();

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
  } else if (const PointerType *DestPTy = destTy->getAs<PointerType>()) {
    srcRecordTy = srcTy->castAs<PointerType>()->getPointeeType();
    destRecordTy = DestPTy->getPointeeType();
  } else {
    srcRecordTy = srcTy;
    destRecordTy = destTy->castAs<ReferenceType>()->getPointeeType();
  }

  assert(srcRecordTy->isRecordType() && "source type must be a record type!");
  emitTypeCheck(TCK_DynamicOperation, DCE->getExprLoc(), ThisAddr.getPointer(),
                srcRecordTy);

  if (DCE->isAlwaysNull())
    return emitDynamicCastToNull(*this, loc, destTy);

  auto destCirTy = mlir::cast<cir::PointerType>(convertType(destTy));
  return CGM.getCXXABI().emitDynamicCast(*this, loc, srcRecordTy, destRecordTy,
                                         destCirTy, isRefCast, ThisAddr);
}
