//===- Calls.cpp - Wrapper for all function and method calls ------*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file This file defines CallEvent and its subclasses, which represent path-
/// sensitive instances of different kinds of function and method calls
/// (C, C++, and Objective-C).
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/AST/ParentMap.h"
#include "clang/Analysis/ProgramPoint.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/DynamicTypeMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace ento;

QualType CallEvent::getResultType() const {
  const Expr *E = getOriginExpr();
  assert(E && "Calls without origin expressions do not have results");
  QualType ResultTy = E->getType();

  ASTContext &Ctx = getState()->getStateManager().getContext();

  // A function that returns a reference to 'int' will have a result type
  // of simply 'int'. Check the origin expr's value kind to recover the
  // proper type.
  switch (E->getValueKind()) {
  case VK_LValue:
    ResultTy = Ctx.getLValueReferenceType(ResultTy);
    break;
  case VK_XValue:
    ResultTy = Ctx.getRValueReferenceType(ResultTy);
    break;
  case VK_RValue:
    // No adjustment is necessary.
    break;
  }

  return ResultTy;
}

static bool isCallback(QualType T) {
  // If a parameter is a block or a callback, assume it can modify pointer.
  if (T->isBlockPointerType() ||
      T->isFunctionPointerType() ||
      T->isObjCSelType())
    return true;

  // Check if a callback is passed inside a struct (for both, struct passed by
  // reference and by value). Dig just one level into the struct for now.

  if (T->isAnyPointerType() || T->isReferenceType())
    T = T->getPointeeType();

  if (const RecordType *RT = T->getAsStructureType()) {
    const RecordDecl *RD = RT->getDecl();
    for (const auto *I : RD->fields()) {
      QualType FieldT = I->getType();
      if (FieldT->isBlockPointerType() || FieldT->isFunctionPointerType())
        return true;
    }
  }
  return false;
}

static bool isVoidPointerToNonConst(QualType T) {
  if (const PointerType *PT = T->getAs<PointerType>()) {
    QualType PointeeTy = PT->getPointeeType();
    if (PointeeTy.isConstQualified())
      return false;
    return PointeeTy->isVoidType();
  } else
    return false;
}

bool CallEvent::hasNonNullArgumentsWithType(bool (*Condition)(QualType)) const {
  unsigned NumOfArgs = getNumArgs();

  // If calling using a function pointer, assume the function does not
  // satisfy the callback.
  // TODO: We could check the types of the arguments here.
  if (!getDecl())
    return false;

  unsigned Idx = 0;
  for (CallEvent::param_type_iterator I = param_type_begin(),
                                      E = param_type_end();
       I != E && Idx < NumOfArgs; ++I, ++Idx) {
    if (NumOfArgs <= Idx)
      break;

    // If the parameter is 0, it's harmless.
    if (getArgSVal(Idx).isZeroConstant())
      continue;

    if (Condition(*I))
      return true;
  }
  return false;
}

bool CallEvent::hasNonZeroCallbackArg() const {
  return hasNonNullArgumentsWithType(isCallback);
}

bool CallEvent::hasVoidPointerToNonConstArg() const {
  return hasNonNullArgumentsWithType(isVoidPointerToNonConst);
}

bool CallEvent::isGlobalCFunction(StringRef FunctionName) const {
  const FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(getDecl());
  if (!FD)
    return false;

  return CheckerContext::isCLibraryFunction(FD, FunctionName);
}

/// \brief Returns true if a type is a pointer-to-const or reference-to-const
/// with no further indirection.
static bool isPointerToConst(QualType Ty) {
  QualType PointeeTy = Ty->getPointeeType();
  if (PointeeTy == QualType())
    return false;
  if (!PointeeTy.isConstQualified())
    return false;
  if (PointeeTy->isAnyPointerType())
    return false;
  return true;
}

// Try to retrieve the function declaration and find the function parameter
// types which are pointers/references to a non-pointer const.
// We will not invalidate the corresponding argument regions.
static void findPtrToConstParams(llvm::SmallSet<unsigned, 4> &PreserveArgs,
                                 const CallEvent &Call) {
  unsigned Idx = 0;
  for (CallEvent::param_type_iterator I = Call.param_type_begin(),
                                      E = Call.param_type_end();
       I != E; ++I, ++Idx) {
    if (isPointerToConst(*I))
      PreserveArgs.insert(Idx);
  }
}

ProgramStateRef CallEvent::invalidateRegions(unsigned BlockCount,
                                             ProgramStateRef Orig) const {
  ProgramStateRef Result = (Orig ? Orig : getState());

  // Don't invalidate anything if the callee is marked pure/const.
  if (const Decl *callee = getDecl())
    if (callee->hasAttr<PureAttr>() || callee->hasAttr<ConstAttr>())
      return Result;

  SmallVector<SVal, 8> ValuesToInvalidate;
  RegionAndSymbolInvalidationTraits ETraits;

  getExtraInvalidatedValues(ValuesToInvalidate, &ETraits);

  // Indexes of arguments whose values will be preserved by the call.
  llvm::SmallSet<unsigned, 4> PreserveArgs;
  if (!argumentsMayEscape())
    findPtrToConstParams(PreserveArgs, *this);

  for (unsigned Idx = 0, Count = getNumArgs(); Idx != Count; ++Idx) {
    // Mark this region for invalidation.  We batch invalidate regions
    // below for efficiency.
    if (PreserveArgs.count(Idx))
      if (const MemRegion *MR = getArgSVal(Idx).getAsRegion())
        ETraits.setTrait(MR->getBaseRegion(),
                        RegionAndSymbolInvalidationTraits::TK_PreserveContents);
        // TODO: Factor this out + handle the lower level const pointers.

    ValuesToInvalidate.push_back(getArgSVal(Idx));
  }

  // Invalidate designated regions using the batch invalidation API.
  // NOTE: Even if RegionsToInvalidate is empty, we may still invalidate
  //  global variables.
  return Result->invalidateRegions(ValuesToInvalidate, getOriginExpr(),
                                   BlockCount, getLocationContext(),
                                   /*CausedByPointerEscape*/ true,
                                   /*Symbols=*/nullptr, this, &ETraits);
}

ProgramPoint CallEvent::getProgramPoint(bool IsPreVisit,
                                        const ProgramPointTag *Tag) const {
  if (const Expr *E = getOriginExpr()) {
    if (IsPreVisit)
      return PreStmt(E, getLocationContext(), Tag);
    return PostStmt(E, getLocationContext(), Tag);
  }

  const Decl *D = getDecl();
  assert(D && "Cannot get a program point without a statement or decl");

  SourceLocation Loc = getSourceRange().getBegin();
  if (IsPreVisit)
    return PreImplicitCall(D, Loc, getLocationContext(), Tag);
  return PostImplicitCall(D, Loc, getLocationContext(), Tag);
}

bool CallEvent::isCalled(const CallDescription &CD) const {
  assert(getKind() != CE_ObjCMessage && "Obj-C methods are not supported");
  if (!CD.IsLookupDone) {
    CD.IsLookupDone = true;
    CD.II = &getState()->getStateManager().getContext().Idents.get(CD.FuncName);
  }
  const IdentifierInfo *II = getCalleeIdentifier();
  if (!II || II != CD.II)
    return false;
  return (CD.RequiredArgs == CallDescription::NoArgRequirement ||
          CD.RequiredArgs == getNumArgs());
}

SVal CallEvent::getArgSVal(unsigned Index) const {
  const Expr *ArgE = getArgExpr(Index);
  if (!ArgE)
    return UnknownVal();
  return getSVal(ArgE);
}

SourceRange CallEvent::getArgSourceRange(unsigned Index) const {
  const Expr *ArgE = getArgExpr(Index);
  if (!ArgE)
    return SourceRange();
  return ArgE->getSourceRange();
}

SVal CallEvent::getReturnValue() const {
  const Expr *E = getOriginExpr();
  if (!E)
    return UndefinedVal();
  return getSVal(E);
}

LLVM_DUMP_METHOD void CallEvent::dump() const { dump(llvm::errs()); }

void CallEvent::dump(raw_ostream &Out) const {
  ASTContext &Ctx = getState()->getStateManager().getContext();
  if (const Expr *E = getOriginExpr()) {
    E->printPretty(Out, nullptr, Ctx.getPrintingPolicy());
    Out << "\n";
    return;
  }

  if (const Decl *D = getDecl()) {
    Out << "Call to ";
    D->print(Out, Ctx.getPrintingPolicy());
    return;
  }

  // FIXME: a string representation of the kind would be nice.
  Out << "Unknown call (type " << getKind() << ")";
}


bool CallEvent::isCallStmt(const Stmt *S) {
  return isa<CallExpr>(S) || isa<ObjCMessageExpr>(S)
                          || isa<CXXConstructExpr>(S)
                          || isa<CXXNewExpr>(S);
}

QualType CallEvent::getDeclaredResultType(const Decl *D) {
  assert(D);
  if (const FunctionDecl* FD = dyn_cast<FunctionDecl>(D))
    return FD->getReturnType();
  if (const ObjCMethodDecl* MD = dyn_cast<ObjCMethodDecl>(D))
    return MD->getReturnType();
  if (const BlockDecl *BD = dyn_cast<BlockDecl>(D)) {
    // Blocks are difficult because the return type may not be stored in the
    // BlockDecl itself. The AST should probably be enhanced, but for now we
    // just do what we can.
    // If the block is declared without an explicit argument list, the
    // signature-as-written just includes the return type, not the entire
    // function type.
    // FIXME: All blocks should have signatures-as-written, even if the return
    // type is inferred. (That's signified with a dependent result type.)
    if (const TypeSourceInfo *TSI = BD->getSignatureAsWritten()) {
      QualType Ty = TSI->getType();
      if (const FunctionType *FT = Ty->getAs<FunctionType>())
        Ty = FT->getReturnType();
      if (!Ty->isDependentType())
        return Ty;
    }

    return QualType();
  }

  llvm_unreachable("unknown callable kind");
}

bool CallEvent::isVariadic(const Decl *D) {
  assert(D);

  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D))
    return FD->isVariadic();
  if (const ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D))
    return MD->isVariadic();
  if (const BlockDecl *BD = dyn_cast<BlockDecl>(D))
    return BD->isVariadic();

  llvm_unreachable("unknown callable kind");
}

static void addParameterValuesToBindings(const StackFrameContext *CalleeCtx,
                                         CallEvent::BindingsTy &Bindings,
                                         SValBuilder &SVB,
                                         const CallEvent &Call,
                                         ArrayRef<ParmVarDecl*> parameters) {
  MemRegionManager &MRMgr = SVB.getRegionManager();

  // If the function has fewer parameters than the call has arguments, we simply
  // do not bind any values to them.
  unsigned NumArgs = Call.getNumArgs();
  unsigned Idx = 0;
  ArrayRef<ParmVarDecl*>::iterator I = parameters.begin(), E = parameters.end();
  for (; I != E && Idx < NumArgs; ++I, ++Idx) {
    const ParmVarDecl *ParamDecl = *I;
    assert(ParamDecl && "Formal parameter has no decl?");

    SVal ArgVal = Call.getArgSVal(Idx);
    if (!ArgVal.isUnknown()) {
      Loc ParamLoc = SVB.makeLoc(MRMgr.getVarRegion(ParamDecl, CalleeCtx));
      Bindings.push_back(std::make_pair(ParamLoc, ArgVal));
    }
  }

  // FIXME: Variadic arguments are not handled at all right now.
}

ArrayRef<ParmVarDecl*> AnyFunctionCall::parameters() const {
  const FunctionDecl *D = getDecl();
  if (!D)
    return None;
  return D->parameters();
}

void AnyFunctionCall::getInitialStackFrameContents(
                                        const StackFrameContext *CalleeCtx,
                                        BindingsTy &Bindings) const {
  const FunctionDecl *D = cast<FunctionDecl>(CalleeCtx->getDecl());
  SValBuilder &SVB = getState()->getStateManager().getSValBuilder();
  addParameterValuesToBindings(CalleeCtx, Bindings, SVB, *this,
                               D->parameters());
}

bool AnyFunctionCall::argumentsMayEscape() const {
  if (CallEvent::argumentsMayEscape() || hasVoidPointerToNonConstArg())
    return true;

  const FunctionDecl *D = getDecl();
  if (!D)
    return true;

  const IdentifierInfo *II = D->getIdentifier();
  if (!II)
    return false;

  // This set of "escaping" APIs is

  // - 'int pthread_setspecific(ptheread_key k, const void *)' stores a
  //   value into thread local storage. The value can later be retrieved with
  //   'void *ptheread_getspecific(pthread_key)'. So even thought the
  //   parameter is 'const void *', the region escapes through the call.
  if (II->isStr("pthread_setspecific"))
    return true;

  // - xpc_connection_set_context stores a value which can be retrieved later
  //   with xpc_connection_get_context.
  if (II->isStr("xpc_connection_set_context"))
    return true;

  // - funopen - sets a buffer for future IO calls.
  if (II->isStr("funopen"))
    return true;

  // - __cxa_demangle - can reallocate memory and can return the pointer to
  // the input buffer.
  if (II->isStr("__cxa_demangle"))
    return true;

  StringRef FName = II->getName();

  // - CoreFoundation functions that end with "NoCopy" can free a passed-in
  //   buffer even if it is const.
  if (FName.endswith("NoCopy"))
    return true;

  // - NSXXInsertXX, for example NSMapInsertIfAbsent, since they can
  //   be deallocated by NSMapRemove.
  if (FName.startswith("NS") && (FName.find("Insert") != StringRef::npos))
    return true;

  // - Many CF containers allow objects to escape through custom
  //   allocators/deallocators upon container construction. (PR12101)
  if (FName.startswith("CF") || FName.startswith("CG")) {
    return StrInStrNoCase(FName, "InsertValue")  != StringRef::npos ||
           StrInStrNoCase(FName, "AddValue")     != StringRef::npos ||
           StrInStrNoCase(FName, "SetValue")     != StringRef::npos ||
           StrInStrNoCase(FName, "WithData")     != StringRef::npos ||
           StrInStrNoCase(FName, "AppendValue")  != StringRef::npos ||
           StrInStrNoCase(FName, "SetAttribute") != StringRef::npos;
  }

  return false;
}


const FunctionDecl *SimpleFunctionCall::getDecl() const {
  const FunctionDecl *D = getOriginExpr()->getDirectCallee();
  if (D)
    return D;

  return getSVal(getOriginExpr()->getCallee()).getAsFunctionDecl();
}


const FunctionDecl *CXXInstanceCall::getDecl() const {
  const CallExpr *CE = cast_or_null<CallExpr>(getOriginExpr());
  if (!CE)
    return AnyFunctionCall::getDecl();

  const FunctionDecl *D = CE->getDirectCallee();
  if (D)
    return D;

  return getSVal(CE->getCallee()).getAsFunctionDecl();
}

void CXXInstanceCall::getExtraInvalidatedValues(
    ValueList &Values, RegionAndSymbolInvalidationTraits *ETraits) const {
  SVal ThisVal = getCXXThisVal();
  Values.push_back(ThisVal);

  // Don't invalidate if the method is const and there are no mutable fields.
  if (const CXXMethodDecl *D = cast_or_null<CXXMethodDecl>(getDecl())) {
    if (!D->isConst())
      return;
    // Get the record decl for the class of 'This'. D->getParent() may return a
    // base class decl, rather than the class of the instance which needs to be
    // checked for mutable fields.
    const Expr *Ex = getCXXThisExpr()->ignoreParenBaseCasts();
    const CXXRecordDecl *ParentRecord = Ex->getType()->getAsCXXRecordDecl();
    if (!ParentRecord || ParentRecord->hasMutableFields())
      return;
    // Preserve CXXThis.
    const MemRegion *ThisRegion = ThisVal.getAsRegion();
    if (!ThisRegion)
      return;

    ETraits->setTrait(ThisRegion->getBaseRegion(),
                      RegionAndSymbolInvalidationTraits::TK_PreserveContents);
  }
}

SVal CXXInstanceCall::getCXXThisVal() const {
  const Expr *Base = getCXXThisExpr();
  // FIXME: This doesn't handle an overloaded ->* operator.
  if (!Base)
    return UnknownVal();

  SVal ThisVal = getSVal(Base);
  assert(ThisVal.isUnknownOrUndef() || ThisVal.getAs<Loc>());
  return ThisVal;
}


RuntimeDefinition CXXInstanceCall::getRuntimeDefinition() const {
  // Do we have a decl at all?
  const Decl *D = getDecl();
  if (!D)
    return RuntimeDefinition();

  // If the method is non-virtual, we know we can inline it.
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(D);
  if (!MD->isVirtual())
    return AnyFunctionCall::getRuntimeDefinition();

  // Do we know the implicit 'this' object being called?
  const MemRegion *R = getCXXThisVal().getAsRegion();
  if (!R)
    return RuntimeDefinition();

  // Do we know anything about the type of 'this'?
  DynamicTypeInfo DynType = getDynamicTypeInfo(getState(), R);
  if (!DynType.isValid())
    return RuntimeDefinition();

  // Is the type a C++ class? (This is mostly a defensive check.)
  QualType RegionType = DynType.getType()->getPointeeType();
  assert(!RegionType.isNull() && "DynamicTypeInfo should always be a pointer.");

  const CXXRecordDecl *RD = RegionType->getAsCXXRecordDecl();
  if (!RD || !RD->hasDefinition())
    return RuntimeDefinition();

  // Find the decl for this method in that class.
  const CXXMethodDecl *Result = MD->getCorrespondingMethodInClass(RD, true);
  if (!Result) {
    // We might not even get the original statically-resolved method due to
    // some particularly nasty casting (e.g. casts to sister classes).
    // However, we should at least be able to search up and down our own class
    // hierarchy, and some real bugs have been caught by checking this.
    assert(!RD->isDerivedFrom(MD->getParent()) && "Couldn't find known method");

    // FIXME: This is checking that our DynamicTypeInfo is at least as good as
    // the static type. However, because we currently don't update
    // DynamicTypeInfo when an object is cast, we can't actually be sure the
    // DynamicTypeInfo is up to date. This assert should be re-enabled once
    // this is fixed. <rdar://problem/12287087>
    //assert(!MD->getParent()->isDerivedFrom(RD) && "Bad DynamicTypeInfo");

    return RuntimeDefinition();
  }

  // Does the decl that we found have an implementation?
  const FunctionDecl *Definition;
  if (!Result->hasBody(Definition))
    return RuntimeDefinition();

  // We found a definition. If we're not sure that this devirtualization is
  // actually what will happen at runtime, make sure to provide the region so
  // that ExprEngine can decide what to do with it.
  if (DynType.canBeASubClass())
    return RuntimeDefinition(Definition, R->StripCasts());
  return RuntimeDefinition(Definition, /*DispatchRegion=*/nullptr);
}

void CXXInstanceCall::getInitialStackFrameContents(
                                            const StackFrameContext *CalleeCtx,
                                            BindingsTy &Bindings) const {
  AnyFunctionCall::getInitialStackFrameContents(CalleeCtx, Bindings);

  // Handle the binding of 'this' in the new stack frame.
  SVal ThisVal = getCXXThisVal();
  if (!ThisVal.isUnknown()) {
    ProgramStateManager &StateMgr = getState()->getStateManager();
    SValBuilder &SVB = StateMgr.getSValBuilder();

    const CXXMethodDecl *MD = cast<CXXMethodDecl>(CalleeCtx->getDecl());
    Loc ThisLoc = SVB.getCXXThis(MD, CalleeCtx);

    // If we devirtualized to a different member function, we need to make sure
    // we have the proper layering of CXXBaseObjectRegions.
    if (MD->getCanonicalDecl() != getDecl()->getCanonicalDecl()) {
      ASTContext &Ctx = SVB.getContext();
      const CXXRecordDecl *Class = MD->getParent();
      QualType Ty = Ctx.getPointerType(Ctx.getRecordType(Class));

      // FIXME: CallEvent maybe shouldn't be directly accessing StoreManager.
      bool Failed;
      ThisVal = StateMgr.getStoreManager().attemptDownCast(ThisVal, Ty, Failed);
      assert(!Failed && "Calling an incorrectly devirtualized method");
    }

    if (!ThisVal.isUnknown())
      Bindings.push_back(std::make_pair(ThisLoc, ThisVal));
  }
}



const Expr *CXXMemberCall::getCXXThisExpr() const {
  return getOriginExpr()->getImplicitObjectArgument();
}

RuntimeDefinition CXXMemberCall::getRuntimeDefinition() const {
  // C++11 [expr.call]p1: ...If the selected function is non-virtual, or if the
  // id-expression in the class member access expression is a qualified-id,
  // that function is called. Otherwise, its final overrider in the dynamic type
  // of the object expression is called.
  if (const MemberExpr *ME = dyn_cast<MemberExpr>(getOriginExpr()->getCallee()))
    if (ME->hasQualifier())
      return AnyFunctionCall::getRuntimeDefinition();

  return CXXInstanceCall::getRuntimeDefinition();
}


const Expr *CXXMemberOperatorCall::getCXXThisExpr() const {
  return getOriginExpr()->getArg(0);
}


const BlockDataRegion *BlockCall::getBlockRegion() const {
  const Expr *Callee = getOriginExpr()->getCallee();
  const MemRegion *DataReg = getSVal(Callee).getAsRegion();

  return dyn_cast_or_null<BlockDataRegion>(DataReg);
}

ArrayRef<ParmVarDecl*> BlockCall::parameters() const {
  const BlockDecl *D = getDecl();
  if (!D)
    return nullptr;
  return D->parameters();
}

void BlockCall::getExtraInvalidatedValues(ValueList &Values,
                  RegionAndSymbolInvalidationTraits *ETraits) const {
  // FIXME: This also needs to invalidate captured globals.
  if (const MemRegion *R = getBlockRegion())
    Values.push_back(loc::MemRegionVal(R));
}

void BlockCall::getInitialStackFrameContents(const StackFrameContext *CalleeCtx,
                                             BindingsTy &Bindings) const {
  SValBuilder &SVB = getState()->getStateManager().getSValBuilder();
  ArrayRef<ParmVarDecl*> Params;
  if (isConversionFromLambda()) {
    auto *LambdaOperatorDecl = cast<CXXMethodDecl>(CalleeCtx->getDecl());
    Params = LambdaOperatorDecl->parameters();

    // For blocks converted from a C++ lambda, the callee declaration is the
    // operator() method on the lambda so we bind "this" to
    // the lambda captured by the block.
    const VarRegion *CapturedLambdaRegion = getRegionStoringCapturedLambda();
    SVal ThisVal = loc::MemRegionVal(CapturedLambdaRegion);
    Loc ThisLoc = SVB.getCXXThis(LambdaOperatorDecl, CalleeCtx);
    Bindings.push_back(std::make_pair(ThisLoc, ThisVal));
  } else {
    Params = cast<BlockDecl>(CalleeCtx->getDecl())->parameters();
  }

  addParameterValuesToBindings(CalleeCtx, Bindings, SVB, *this,
                               Params);
}


SVal CXXConstructorCall::getCXXThisVal() const {
  if (Data)
    return loc::MemRegionVal(static_cast<const MemRegion *>(Data));
  return UnknownVal();
}

void CXXConstructorCall::getExtraInvalidatedValues(ValueList &Values,
                           RegionAndSymbolInvalidationTraits *ETraits) const {
  if (Data)
    Values.push_back(loc::MemRegionVal(static_cast<const MemRegion *>(Data)));
}

void CXXConstructorCall::getInitialStackFrameContents(
                                             const StackFrameContext *CalleeCtx,
                                             BindingsTy &Bindings) const {
  AnyFunctionCall::getInitialStackFrameContents(CalleeCtx, Bindings);

  SVal ThisVal = getCXXThisVal();
  if (!ThisVal.isUnknown()) {
    SValBuilder &SVB = getState()->getStateManager().getSValBuilder();
    const CXXMethodDecl *MD = cast<CXXMethodDecl>(CalleeCtx->getDecl());
    Loc ThisLoc = SVB.getCXXThis(MD, CalleeCtx);
    Bindings.push_back(std::make_pair(ThisLoc, ThisVal));
  }
}

SVal CXXDestructorCall::getCXXThisVal() const {
  if (Data)
    return loc::MemRegionVal(DtorDataTy::getFromOpaqueValue(Data).getPointer());
  return UnknownVal();
}

RuntimeDefinition CXXDestructorCall::getRuntimeDefinition() const {
  // Base destructors are always called non-virtually.
  // Skip CXXInstanceCall's devirtualization logic in this case.
  if (isBaseDestructor())
    return AnyFunctionCall::getRuntimeDefinition();

  return CXXInstanceCall::getRuntimeDefinition();
}

ArrayRef<ParmVarDecl*> ObjCMethodCall::parameters() const {
  const ObjCMethodDecl *D = getDecl();
  if (!D)
    return None;
  return D->parameters();
}

void ObjCMethodCall::getExtraInvalidatedValues(
    ValueList &Values, RegionAndSymbolInvalidationTraits *ETraits) const {

  // If the method call is a setter for property known to be backed by
  // an instance variable, don't invalidate the entire receiver, just
  // the storage for that instance variable.
  if (const ObjCPropertyDecl *PropDecl = getAccessedProperty()) {
    if (const ObjCIvarDecl *PropIvar = PropDecl->getPropertyIvarDecl()) {
      SVal IvarLVal = getState()->getLValue(PropIvar, getReceiverSVal());
      const MemRegion *IvarRegion = IvarLVal.getAsRegion();
      ETraits->setTrait(
          IvarRegion,
          RegionAndSymbolInvalidationTraits::TK_DoNotInvalidateSuperRegion);
      ETraits->setTrait(IvarRegion,
                        RegionAndSymbolInvalidationTraits::TK_SuppressEscape);
      Values.push_back(IvarLVal);
      return;
    }
  }

  Values.push_back(getReceiverSVal());
}

SVal ObjCMethodCall::getSelfSVal() const {
  const LocationContext *LCtx = getLocationContext();
  const ImplicitParamDecl *SelfDecl = LCtx->getSelfDecl();
  if (!SelfDecl)
    return SVal();
  return getState()->getSVal(getState()->getRegion(SelfDecl, LCtx));
}

SVal ObjCMethodCall::getReceiverSVal() const {
  // FIXME: Is this the best way to handle class receivers?
  if (!isInstanceMessage())
    return UnknownVal();

  if (const Expr *RecE = getOriginExpr()->getInstanceReceiver())
    return getSVal(RecE);

  // An instance message with no expression means we are sending to super.
  // In this case the object reference is the same as 'self'.
  assert(getOriginExpr()->getReceiverKind() == ObjCMessageExpr::SuperInstance);
  SVal SelfVal = getSelfSVal();
  assert(SelfVal.isValid() && "Calling super but not in ObjC method");
  return SelfVal;
}

bool ObjCMethodCall::isReceiverSelfOrSuper() const {
  if (getOriginExpr()->getReceiverKind() == ObjCMessageExpr::SuperInstance ||
      getOriginExpr()->getReceiverKind() == ObjCMessageExpr::SuperClass)
      return true;

  if (!isInstanceMessage())
    return false;

  SVal RecVal = getSVal(getOriginExpr()->getInstanceReceiver());

  return (RecVal == getSelfSVal());
}

SourceRange ObjCMethodCall::getSourceRange() const {
  switch (getMessageKind()) {
  case OCM_Message:
    return getOriginExpr()->getSourceRange();
  case OCM_PropertyAccess:
  case OCM_Subscript:
    return getContainingPseudoObjectExpr()->getSourceRange();
  }
  llvm_unreachable("unknown message kind");
}

typedef llvm::PointerIntPair<const PseudoObjectExpr *, 2> ObjCMessageDataTy;

const PseudoObjectExpr *ObjCMethodCall::getContainingPseudoObjectExpr() const {
  assert(Data && "Lazy lookup not yet performed.");
  assert(getMessageKind() != OCM_Message && "Explicit message send.");
  return ObjCMessageDataTy::getFromOpaqueValue(Data).getPointer();
}

static const Expr *
getSyntacticFromForPseudoObjectExpr(const PseudoObjectExpr *POE) {
  const Expr *Syntactic = POE->getSyntacticForm();

  // This handles the funny case of assigning to the result of a getter.
  // This can happen if the getter returns a non-const reference.
  if (const BinaryOperator *BO = dyn_cast<BinaryOperator>(Syntactic))
    Syntactic = BO->getLHS();

  return Syntactic;
}

ObjCMessageKind ObjCMethodCall::getMessageKind() const {
  if (!Data) {

    // Find the parent, ignoring implicit casts.
    ParentMap &PM = getLocationContext()->getParentMap();
    const Stmt *S = PM.getParentIgnoreParenCasts(getOriginExpr());

    // Check if parent is a PseudoObjectExpr.
    if (const PseudoObjectExpr *POE = dyn_cast_or_null<PseudoObjectExpr>(S)) {
      const Expr *Syntactic = getSyntacticFromForPseudoObjectExpr(POE);

      ObjCMessageKind K;
      switch (Syntactic->getStmtClass()) {
      case Stmt::ObjCPropertyRefExprClass:
        K = OCM_PropertyAccess;
        break;
      case Stmt::ObjCSubscriptRefExprClass:
        K = OCM_Subscript;
        break;
      default:
        // FIXME: Can this ever happen?
        K = OCM_Message;
        break;
      }

      if (K != OCM_Message) {
        const_cast<ObjCMethodCall *>(this)->Data
          = ObjCMessageDataTy(POE, K).getOpaqueValue();
        assert(getMessageKind() == K);
        return K;
      }
    }

    const_cast<ObjCMethodCall *>(this)->Data
      = ObjCMessageDataTy(nullptr, 1).getOpaqueValue();
    assert(getMessageKind() == OCM_Message);
    return OCM_Message;
  }

  ObjCMessageDataTy Info = ObjCMessageDataTy::getFromOpaqueValue(Data);
  if (!Info.getPointer())
    return OCM_Message;
  return static_cast<ObjCMessageKind>(Info.getInt());
}

const ObjCPropertyDecl *ObjCMethodCall::getAccessedProperty() const {
  // Look for properties accessed with property syntax (foo.bar = ...)
  if ( getMessageKind() == OCM_PropertyAccess) {
    const PseudoObjectExpr *POE = getContainingPseudoObjectExpr();
    assert(POE && "Property access without PseudoObjectExpr?");

    const Expr *Syntactic = getSyntacticFromForPseudoObjectExpr(POE);
    auto *RefExpr = cast<ObjCPropertyRefExpr>(Syntactic);

    if (RefExpr->isExplicitProperty())
      return RefExpr->getExplicitProperty();
  }

  // Look for properties accessed with method syntax ([foo setBar:...]).
  const ObjCMethodDecl *MD = getDecl();
  if (!MD || !MD->isPropertyAccessor())
    return nullptr;

  // Note: This is potentially quite slow.
  return MD->findPropertyDecl();
}

bool ObjCMethodCall::canBeOverridenInSubclass(ObjCInterfaceDecl *IDecl,
                                             Selector Sel) const {
  assert(IDecl);
  const SourceManager &SM =
    getState()->getStateManager().getContext().getSourceManager();

  // If the class interface is declared inside the main file, assume it is not
  // subcassed.
  // TODO: It could actually be subclassed if the subclass is private as well.
  // This is probably very rare.
  SourceLocation InterfLoc = IDecl->getEndOfDefinitionLoc();
  if (InterfLoc.isValid() && SM.isInMainFile(InterfLoc))
    return false;

  // Assume that property accessors are not overridden.
  if (getMessageKind() == OCM_PropertyAccess)
    return false;

  // We assume that if the method is public (declared outside of main file) or
  // has a parent which publicly declares the method, the method could be
  // overridden in a subclass.

  // Find the first declaration in the class hierarchy that declares
  // the selector.
  ObjCMethodDecl *D = nullptr;
  while (true) {
    D = IDecl->lookupMethod(Sel, true);

    // Cannot find a public definition.
    if (!D)
      return false;

    // If outside the main file,
    if (D->getLocation().isValid() && !SM.isInMainFile(D->getLocation()))
      return true;

    if (D->isOverriding()) {
      // Search in the superclass on the next iteration.
      IDecl = D->getClassInterface();
      if (!IDecl)
        return false;

      IDecl = IDecl->getSuperClass();
      if (!IDecl)
        return false;

      continue;
    }

    return false;
  };

  llvm_unreachable("The while loop should always terminate.");
}

static const ObjCMethodDecl *findDefiningRedecl(const ObjCMethodDecl *MD) {
  if (!MD)
    return MD;

  // Find the redeclaration that defines the method.
  if (!MD->hasBody()) {
    for (auto I : MD->redecls())
      if (I->hasBody())
        MD = cast<ObjCMethodDecl>(I);
  }
  return MD;
}

static bool isCallToSelfClass(const ObjCMessageExpr *ME) {
  const Expr* InstRec = ME->getInstanceReceiver();
  if (!InstRec)
    return false;
  const auto *InstRecIg = dyn_cast<DeclRefExpr>(InstRec->IgnoreParenImpCasts());

  // Check that receiver is called 'self'.
  if (!InstRecIg || !InstRecIg->getFoundDecl() ||
      !InstRecIg->getFoundDecl()->getName().equals("self"))
    return false;

  // Check that the method name is 'class'.
  if (ME->getSelector().getNumArgs() != 0 ||
      !ME->getSelector().getNameForSlot(0).equals("class"))
    return false;

  return true;
}

RuntimeDefinition ObjCMethodCall::getRuntimeDefinition() const {
  const ObjCMessageExpr *E = getOriginExpr();
  assert(E);
  Selector Sel = E->getSelector();

  if (E->isInstanceMessage()) {

    // Find the receiver type.
    const ObjCObjectPointerType *ReceiverT = nullptr;
    bool CanBeSubClassed = false;
    QualType SupersType = E->getSuperType();
    const MemRegion *Receiver = nullptr;

    if (!SupersType.isNull()) {
      // The receiver is guaranteed to be 'super' in this case.
      // Super always means the type of immediate predecessor to the method
      // where the call occurs.
      ReceiverT = cast<ObjCObjectPointerType>(SupersType);
    } else {
      Receiver = getReceiverSVal().getAsRegion();
      if (!Receiver)
        return RuntimeDefinition();

      DynamicTypeInfo DTI = getDynamicTypeInfo(getState(), Receiver);
      QualType DynType = DTI.getType();
      CanBeSubClassed = DTI.canBeASubClass();
      ReceiverT = dyn_cast<ObjCObjectPointerType>(DynType.getCanonicalType());

      if (ReceiverT && CanBeSubClassed)
        if (ObjCInterfaceDecl *IDecl = ReceiverT->getInterfaceDecl())
          if (!canBeOverridenInSubclass(IDecl, Sel))
            CanBeSubClassed = false;
    }

    // Handle special cases of '[self classMethod]' and
    // '[[self class] classMethod]', which are treated by the compiler as
    // instance (not class) messages. We will statically dispatch to those.
    if (auto *PT = dyn_cast_or_null<ObjCObjectPointerType>(ReceiverT)) {
      // For [self classMethod], return the compiler visible declaration.
      if (PT->getObjectType()->isObjCClass() &&
          Receiver == getSelfSVal().getAsRegion())
        return RuntimeDefinition(findDefiningRedecl(E->getMethodDecl()));

      // Similarly, handle [[self class] classMethod].
      // TODO: We are currently doing a syntactic match for this pattern with is
      // limiting as the test cases in Analysis/inlining/InlineObjCClassMethod.m
      // shows. A better way would be to associate the meta type with the symbol
      // using the dynamic type info tracking and use it here. We can add a new
      // SVal for ObjC 'Class' values that know what interface declaration they
      // come from. Then 'self' in a class method would be filled in with
      // something meaningful in ObjCMethodCall::getReceiverSVal() and we could
      // do proper dynamic dispatch for class methods just like we do for
      // instance methods now.
      if (E->getInstanceReceiver())
        if (const auto *M = dyn_cast<ObjCMessageExpr>(E->getInstanceReceiver()))
          if (isCallToSelfClass(M))
            return RuntimeDefinition(findDefiningRedecl(E->getMethodDecl()));
    }

    // Lookup the instance method implementation.
    if (ReceiverT)
      if (ObjCInterfaceDecl *IDecl = ReceiverT->getInterfaceDecl()) {
        // Repeatedly calling lookupPrivateMethod() is expensive, especially
        // when in many cases it returns null.  We cache the results so
        // that repeated queries on the same ObjCIntefaceDecl and Selector
        // don't incur the same cost.  On some test cases, we can see the
        // same query being issued thousands of times.
        //
        // NOTE: This cache is essentially a "global" variable, but it
        // only gets lazily created when we get here.  The value of the
        // cache probably comes from it being global across ExprEngines,
        // where the same queries may get issued.  If we are worried about
        // concurrency, or possibly loading/unloading ASTs, etc., we may
        // need to revisit this someday.  In terms of memory, this table
        // stays around until clang quits, which also may be bad if we
        // need to release memory.
        typedef std::pair<const ObjCInterfaceDecl*, Selector>
                PrivateMethodKey;
        typedef llvm::DenseMap<PrivateMethodKey,
                               Optional<const ObjCMethodDecl *> >
                PrivateMethodCache;

        static PrivateMethodCache PMC;
        Optional<const ObjCMethodDecl *> &Val = PMC[std::make_pair(IDecl, Sel)];

        // Query lookupPrivateMethod() if the cache does not hit.
        if (!Val.hasValue()) {
          Val = IDecl->lookupPrivateMethod(Sel);

          // If the method is a property accessor, we should try to "inline" it
          // even if we don't actually have an implementation.
          if (!*Val)
            if (const ObjCMethodDecl *CompileTimeMD = E->getMethodDecl())
              if (CompileTimeMD->isPropertyAccessor()) {
                if (!CompileTimeMD->getSelfDecl() &&
                    isa<ObjCCategoryDecl>(CompileTimeMD->getDeclContext())) {
                  // If the method is an accessor in a category, and it doesn't
                  // have a self declaration, first
                  // try to find the method in a class extension. This
                  // works around a bug in Sema where multiple accessors
                  // are synthesized for properties in class
                  // extensions that are redeclared in a category and the
                  // the implicit parameters are not filled in for
                  // the method on the category.
                  // This ensures we find the accessor in the extension, which
                  // has the implicit parameters filled in.
                  auto *ID = CompileTimeMD->getClassInterface();
                  for (auto *CatDecl : ID->visible_extensions()) {
                    Val = CatDecl->getMethod(Sel,
                                             CompileTimeMD->isInstanceMethod());
                    if (*Val)
                      break;
                  }
                }
                if (!*Val)
                  Val = IDecl->lookupInstanceMethod(Sel);
              }
        }

        const ObjCMethodDecl *MD = Val.getValue();
        if (CanBeSubClassed)
          return RuntimeDefinition(MD, Receiver);
        else
          return RuntimeDefinition(MD, nullptr);
      }

  } else {
    // This is a class method.
    // If we have type info for the receiver class, we are calling via
    // class name.
    if (ObjCInterfaceDecl *IDecl = E->getReceiverInterface()) {
      // Find/Return the method implementation.
      return RuntimeDefinition(IDecl->lookupPrivateClassMethod(Sel));
    }
  }

  return RuntimeDefinition();
}

bool ObjCMethodCall::argumentsMayEscape() const {
  if (isInSystemHeader() && !isInstanceMessage()) {
    Selector Sel = getSelector();
    if (Sel.getNumArgs() == 1 &&
        Sel.getIdentifierInfoForSlot(0)->isStr("valueWithPointer"))
      return true;
  }

  return CallEvent::argumentsMayEscape();
}

void ObjCMethodCall::getInitialStackFrameContents(
                                             const StackFrameContext *CalleeCtx,
                                             BindingsTy &Bindings) const {
  const ObjCMethodDecl *D = cast<ObjCMethodDecl>(CalleeCtx->getDecl());
  SValBuilder &SVB = getState()->getStateManager().getSValBuilder();
  addParameterValuesToBindings(CalleeCtx, Bindings, SVB, *this,
                               D->parameters());

  SVal SelfVal = getReceiverSVal();
  if (!SelfVal.isUnknown()) {
    const VarDecl *SelfD = CalleeCtx->getAnalysisDeclContext()->getSelfDecl();
    MemRegionManager &MRMgr = SVB.getRegionManager();
    Loc SelfLoc = SVB.makeLoc(MRMgr.getVarRegion(SelfD, CalleeCtx));
    Bindings.push_back(std::make_pair(SelfLoc, SelfVal));
  }
}

CallEventRef<>
CallEventManager::getSimpleCall(const CallExpr *CE, ProgramStateRef State,
                                const LocationContext *LCtx) {
  if (const CXXMemberCallExpr *MCE = dyn_cast<CXXMemberCallExpr>(CE))
    return create<CXXMemberCall>(MCE, State, LCtx);

  if (const CXXOperatorCallExpr *OpCE = dyn_cast<CXXOperatorCallExpr>(CE)) {
    const FunctionDecl *DirectCallee = OpCE->getDirectCallee();
    if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(DirectCallee))
      if (MD->isInstance())
        return create<CXXMemberOperatorCall>(OpCE, State, LCtx);

  } else if (CE->getCallee()->getType()->isBlockPointerType()) {
    return create<BlockCall>(CE, State, LCtx);
  }

  // Otherwise, it's a normal function call, static member function call, or
  // something we can't reason about.
  return create<SimpleFunctionCall>(CE, State, LCtx);
}


CallEventRef<>
CallEventManager::getCaller(const StackFrameContext *CalleeCtx,
                            ProgramStateRef State) {
  const LocationContext *ParentCtx = CalleeCtx->getParent();
  const LocationContext *CallerCtx = ParentCtx->getCurrentStackFrame();
  assert(CallerCtx && "This should not be used for top-level stack frames");

  const Stmt *CallSite = CalleeCtx->getCallSite();

  if (CallSite) {
    if (const CallExpr *CE = dyn_cast<CallExpr>(CallSite))
      return getSimpleCall(CE, State, CallerCtx);

    switch (CallSite->getStmtClass()) {
    case Stmt::CXXConstructExprClass:
    case Stmt::CXXTemporaryObjectExprClass: {
      SValBuilder &SVB = State->getStateManager().getSValBuilder();
      const CXXMethodDecl *Ctor = cast<CXXMethodDecl>(CalleeCtx->getDecl());
      Loc ThisPtr = SVB.getCXXThis(Ctor, CalleeCtx);
      SVal ThisVal = State->getSVal(ThisPtr);

      return getCXXConstructorCall(cast<CXXConstructExpr>(CallSite),
                                   ThisVal.getAsRegion(), State, CallerCtx);
    }
    case Stmt::CXXNewExprClass:
      return getCXXAllocatorCall(cast<CXXNewExpr>(CallSite), State, CallerCtx);
    case Stmt::ObjCMessageExprClass:
      return getObjCMethodCall(cast<ObjCMessageExpr>(CallSite),
                               State, CallerCtx);
    default:
      llvm_unreachable("This is not an inlineable statement.");
    }
  }

  // Fall back to the CFG. The only thing we haven't handled yet is
  // destructors, though this could change in the future.
  const CFGBlock *B = CalleeCtx->getCallSiteBlock();
  CFGElement E = (*B)[CalleeCtx->getIndex()];
  assert(E.getAs<CFGImplicitDtor>() &&
         "All other CFG elements should have exprs");
  assert(!E.getAs<CFGTemporaryDtor>() && "We don't handle temporaries yet");

  SValBuilder &SVB = State->getStateManager().getSValBuilder();
  const CXXDestructorDecl *Dtor = cast<CXXDestructorDecl>(CalleeCtx->getDecl());
  Loc ThisPtr = SVB.getCXXThis(Dtor, CalleeCtx);
  SVal ThisVal = State->getSVal(ThisPtr);

  const Stmt *Trigger;
  if (Optional<CFGAutomaticObjDtor> AutoDtor = E.getAs<CFGAutomaticObjDtor>())
    Trigger = AutoDtor->getTriggerStmt();
  else if (Optional<CFGDeleteDtor> DeleteDtor = E.getAs<CFGDeleteDtor>())
    Trigger = cast<Stmt>(DeleteDtor->getDeleteExpr());
  else
    Trigger = Dtor->getBody();

  return getCXXDestructorCall(Dtor, Trigger, ThisVal.getAsRegion(),
                              E.getAs<CFGBaseDtor>().hasValue(), State,
                              CallerCtx);
}
