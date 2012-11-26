//===--- CGException.cpp - Emit LLVM Code for C++ exceptions --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ exception related code generation.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "CGCleanup.h"
#include "CGObjCRuntime.h"
#include "TargetInfo.h"
#include "clang/AST/StmtCXX.h"
#include "llvm/Intrinsics.h"
#include "llvm/Support/CallSite.h"

using namespace clang;
using namespace CodeGen;

static llvm::Constant *getAllocateExceptionFn(CodeGenFunction &CGF) {
  // void *__cxa_allocate_exception(size_t thrown_size);

  llvm::FunctionType *FTy =
    llvm::FunctionType::get(CGF.Int8PtrTy, CGF.SizeTy, /*IsVarArgs=*/false);

  return CGF.CGM.CreateRuntimeFunction(FTy, "__cxa_allocate_exception");
}

static llvm::Constant *getFreeExceptionFn(CodeGenFunction &CGF) {
  // void __cxa_free_exception(void *thrown_exception);

  llvm::FunctionType *FTy =
    llvm::FunctionType::get(CGF.VoidTy, CGF.Int8PtrTy, /*IsVarArgs=*/false);

  return CGF.CGM.CreateRuntimeFunction(FTy, "__cxa_free_exception");
}

static llvm::Constant *getThrowFn(CodeGenFunction &CGF) {
  // void __cxa_throw(void *thrown_exception, std::type_info *tinfo,
  //                  void (*dest) (void *));

  llvm::Type *Args[3] = { CGF.Int8PtrTy, CGF.Int8PtrTy, CGF.Int8PtrTy };
  llvm::FunctionType *FTy =
    llvm::FunctionType::get(CGF.VoidTy, Args, /*IsVarArgs=*/false);

  return CGF.CGM.CreateRuntimeFunction(FTy, "__cxa_throw");
}

static llvm::Constant *getReThrowFn(CodeGenFunction &CGF) {
  // void __cxa_rethrow();

  llvm::FunctionType *FTy =
    llvm::FunctionType::get(CGF.VoidTy, /*IsVarArgs=*/false);

  return CGF.CGM.CreateRuntimeFunction(FTy, "__cxa_rethrow");
}

static llvm::Constant *getGetExceptionPtrFn(CodeGenFunction &CGF) {
  // void *__cxa_get_exception_ptr(void*);

  llvm::FunctionType *FTy =
    llvm::FunctionType::get(CGF.Int8PtrTy, CGF.Int8PtrTy, /*IsVarArgs=*/false);

  return CGF.CGM.CreateRuntimeFunction(FTy, "__cxa_get_exception_ptr");
}

static llvm::Constant *getBeginCatchFn(CodeGenFunction &CGF) {
  // void *__cxa_begin_catch(void*);

  llvm::FunctionType *FTy =
    llvm::FunctionType::get(CGF.Int8PtrTy, CGF.Int8PtrTy, /*IsVarArgs=*/false);

  return CGF.CGM.CreateRuntimeFunction(FTy, "__cxa_begin_catch");
}

static llvm::Constant *getEndCatchFn(CodeGenFunction &CGF) {
  // void __cxa_end_catch();

  llvm::FunctionType *FTy =
    llvm::FunctionType::get(CGF.VoidTy, /*IsVarArgs=*/false);

  return CGF.CGM.CreateRuntimeFunction(FTy, "__cxa_end_catch");
}

static llvm::Constant *getUnexpectedFn(CodeGenFunction &CGF) {
  // void __cxa_call_unexepcted(void *thrown_exception);

  llvm::FunctionType *FTy =
    llvm::FunctionType::get(CGF.VoidTy, CGF.Int8PtrTy, /*IsVarArgs=*/false);

  return CGF.CGM.CreateRuntimeFunction(FTy, "__cxa_call_unexpected");
}

llvm::Constant *CodeGenFunction::getUnwindResumeFn() {
  llvm::FunctionType *FTy =
    llvm::FunctionType::get(VoidTy, Int8PtrTy, /*IsVarArgs=*/false);

  if (CGM.getLangOpts().SjLjExceptions)
    return CGM.CreateRuntimeFunction(FTy, "_Unwind_SjLj_Resume");
  return CGM.CreateRuntimeFunction(FTy, "_Unwind_Resume");
}

llvm::Constant *CodeGenFunction::getUnwindResumeOrRethrowFn() {
  llvm::FunctionType *FTy =
    llvm::FunctionType::get(VoidTy, Int8PtrTy, /*IsVarArgs=*/false);

  if (CGM.getLangOpts().SjLjExceptions)
    return CGM.CreateRuntimeFunction(FTy, "_Unwind_SjLj_Resume_or_Rethrow");
  return CGM.CreateRuntimeFunction(FTy, "_Unwind_Resume_or_Rethrow");
}

static llvm::Constant *getTerminateFn(CodeGenFunction &CGF) {
  // void __terminate();

  llvm::FunctionType *FTy =
    llvm::FunctionType::get(CGF.VoidTy, /*IsVarArgs=*/false);

  StringRef name;

  // In C++, use std::terminate().
  if (CGF.getLangOpts().CPlusPlus)
    name = "_ZSt9terminatev"; // FIXME: mangling!
  else if (CGF.getLangOpts().ObjC1 &&
           CGF.getLangOpts().ObjCRuntime.hasTerminate())
    name = "objc_terminate";
  else
    name = "abort";
  return CGF.CGM.CreateRuntimeFunction(FTy, name);
}

static llvm::Constant *getCatchallRethrowFn(CodeGenFunction &CGF,
                                            StringRef Name) {
  llvm::FunctionType *FTy =
    llvm::FunctionType::get(CGF.VoidTy, CGF.Int8PtrTy, /*IsVarArgs=*/false);

  return CGF.CGM.CreateRuntimeFunction(FTy, Name);
}

namespace {
  /// The exceptions personality for a function.
  struct EHPersonality {
    const char *PersonalityFn;

    // If this is non-null, this personality requires a non-standard
    // function for rethrowing an exception after a catchall cleanup.
    // This function must have prototype void(void*).
    const char *CatchallRethrowFn;

    static const EHPersonality &get(const LangOptions &Lang);
    static const EHPersonality GNU_C;
    static const EHPersonality GNU_C_SJLJ;
    static const EHPersonality GNU_ObjC;
    static const EHPersonality GNU_ObjCXX;
    static const EHPersonality NeXT_ObjC;
    static const EHPersonality GNU_CPlusPlus;
    static const EHPersonality GNU_CPlusPlus_SJLJ;
  };
}

const EHPersonality EHPersonality::GNU_C = { "__gcc_personality_v0", 0 };
const EHPersonality EHPersonality::GNU_C_SJLJ = { "__gcc_personality_sj0", 0 };
const EHPersonality EHPersonality::NeXT_ObjC = { "__objc_personality_v0", 0 };
const EHPersonality EHPersonality::GNU_CPlusPlus = { "__gxx_personality_v0", 0};
const EHPersonality
EHPersonality::GNU_CPlusPlus_SJLJ = { "__gxx_personality_sj0", 0 };
const EHPersonality
EHPersonality::GNU_ObjC = {"__gnu_objc_personality_v0", "objc_exception_throw"};
const EHPersonality
EHPersonality::GNU_ObjCXX = { "__gnustep_objcxx_personality_v0", 0 };

static const EHPersonality &getCPersonality(const LangOptions &L) {
  if (L.SjLjExceptions)
    return EHPersonality::GNU_C_SJLJ;
  return EHPersonality::GNU_C;
}

static const EHPersonality &getObjCPersonality(const LangOptions &L) {
  switch (L.ObjCRuntime.getKind()) {
  case ObjCRuntime::FragileMacOSX:
    return getCPersonality(L);
  case ObjCRuntime::MacOSX:
  case ObjCRuntime::iOS:
    return EHPersonality::NeXT_ObjC;
  case ObjCRuntime::GNUstep:
  case ObjCRuntime::GCC:
  case ObjCRuntime::ObjFW:
    return EHPersonality::GNU_ObjC;
  }
  llvm_unreachable("bad runtime kind");
}

static const EHPersonality &getCXXPersonality(const LangOptions &L) {
  if (L.SjLjExceptions)
    return EHPersonality::GNU_CPlusPlus_SJLJ;
  else
    return EHPersonality::GNU_CPlusPlus;
}

/// Determines the personality function to use when both C++
/// and Objective-C exceptions are being caught.
static const EHPersonality &getObjCXXPersonality(const LangOptions &L) {
  switch (L.ObjCRuntime.getKind()) {
  // The ObjC personality defers to the C++ personality for non-ObjC
  // handlers.  Unlike the C++ case, we use the same personality
  // function on targets using (backend-driven) SJLJ EH.
  case ObjCRuntime::MacOSX:
  case ObjCRuntime::iOS:
    return EHPersonality::NeXT_ObjC;

  // In the fragile ABI, just use C++ exception handling and hope
  // they're not doing crazy exception mixing.
  case ObjCRuntime::FragileMacOSX:
    return getCXXPersonality(L);

  // The GCC runtime's personality function inherently doesn't support
  // mixed EH.  Use the C++ personality just to avoid returning null.
  case ObjCRuntime::GCC:
  case ObjCRuntime::ObjFW: // XXX: this will change soon
    return EHPersonality::GNU_ObjC;
  case ObjCRuntime::GNUstep:
    return EHPersonality::GNU_ObjCXX;
  }
  llvm_unreachable("bad runtime kind");
}

const EHPersonality &EHPersonality::get(const LangOptions &L) {
  if (L.CPlusPlus && L.ObjC1)
    return getObjCXXPersonality(L);
  else if (L.CPlusPlus)
    return getCXXPersonality(L);
  else if (L.ObjC1)
    return getObjCPersonality(L);
  else
    return getCPersonality(L);
}

static llvm::Constant *getPersonalityFn(CodeGenModule &CGM,
                                        const EHPersonality &Personality) {
  llvm::Constant *Fn =
    CGM.CreateRuntimeFunction(llvm::FunctionType::get(CGM.Int32Ty, true),
                              Personality.PersonalityFn);
  return Fn;
}

static llvm::Constant *getOpaquePersonalityFn(CodeGenModule &CGM,
                                        const EHPersonality &Personality) {
  llvm::Constant *Fn = getPersonalityFn(CGM, Personality);
  return llvm::ConstantExpr::getBitCast(Fn, CGM.Int8PtrTy);
}

/// Check whether a personality function could reasonably be swapped
/// for a C++ personality function.
static bool PersonalityHasOnlyCXXUses(llvm::Constant *Fn) {
  for (llvm::Constant::use_iterator
         I = Fn->use_begin(), E = Fn->use_end(); I != E; ++I) {
    llvm::User *User = *I;

    // Conditionally white-list bitcasts.
    if (llvm::ConstantExpr *CE = dyn_cast<llvm::ConstantExpr>(User)) {
      if (CE->getOpcode() != llvm::Instruction::BitCast) return false;
      if (!PersonalityHasOnlyCXXUses(CE))
        return false;
      continue;
    }

    // Otherwise, it has to be a landingpad instruction.
    llvm::LandingPadInst *LPI = dyn_cast<llvm::LandingPadInst>(User);
    if (!LPI) return false;

    for (unsigned I = 0, E = LPI->getNumClauses(); I != E; ++I) {
      // Look for something that would've been returned by the ObjC
      // runtime's GetEHType() method.
      llvm::Value *Val = LPI->getClause(I)->stripPointerCasts();
      if (LPI->isCatch(I)) {
        // Check if the catch value has the ObjC prefix.
        if (llvm::GlobalVariable *GV = dyn_cast<llvm::GlobalVariable>(Val))
          // ObjC EH selector entries are always global variables with
          // names starting like this.
          if (GV->getName().startswith("OBJC_EHTYPE"))
            return false;
      } else {
        // Check if any of the filter values have the ObjC prefix.
        llvm::Constant *CVal = cast<llvm::Constant>(Val);
        for (llvm::User::op_iterator
               II = CVal->op_begin(), IE = CVal->op_end(); II != IE; ++II) {
          if (llvm::GlobalVariable *GV =
              cast<llvm::GlobalVariable>((*II)->stripPointerCasts()))
            // ObjC EH selector entries are always global variables with
            // names starting like this.
            if (GV->getName().startswith("OBJC_EHTYPE"))
              return false;
        }
      }
    }
  }

  return true;
}

/// Try to use the C++ personality function in ObjC++.  Not doing this
/// can cause some incompatibilities with gcc, which is more
/// aggressive about only using the ObjC++ personality in a function
/// when it really needs it.
void CodeGenModule::SimplifyPersonality() {
  // If we're not in ObjC++ -fexceptions, there's nothing to do.
  if (!LangOpts.CPlusPlus || !LangOpts.ObjC1 || !LangOpts.Exceptions)
    return;

  // Both the problem this endeavors to fix and the way the logic
  // above works is specific to the NeXT runtime.
  if (!LangOpts.ObjCRuntime.isNeXTFamily())
    return;

  const EHPersonality &ObjCXX = EHPersonality::get(LangOpts);
  const EHPersonality &CXX = getCXXPersonality(LangOpts);
  if (&ObjCXX == &CXX)
    return;

  assert(std::strcmp(ObjCXX.PersonalityFn, CXX.PersonalityFn) != 0 &&
         "Different EHPersonalities using the same personality function.");

  llvm::Function *Fn = getModule().getFunction(ObjCXX.PersonalityFn);

  // Nothing to do if it's unused.
  if (!Fn || Fn->use_empty()) return;
  
  // Can't do the optimization if it has non-C++ uses.
  if (!PersonalityHasOnlyCXXUses(Fn)) return;

  // Create the C++ personality function and kill off the old
  // function.
  llvm::Constant *CXXFn = getPersonalityFn(*this, CXX);

  // This can happen if the user is screwing with us.
  if (Fn->getType() != CXXFn->getType()) return;

  Fn->replaceAllUsesWith(CXXFn);
  Fn->eraseFromParent();
}

/// Returns the value to inject into a selector to indicate the
/// presence of a catch-all.
static llvm::Constant *getCatchAllValue(CodeGenFunction &CGF) {
  // Possibly we should use @llvm.eh.catch.all.value here.
  return llvm::ConstantPointerNull::get(CGF.Int8PtrTy);
}

namespace {
  /// A cleanup to free the exception object if its initialization
  /// throws.
  struct FreeException : EHScopeStack::Cleanup {
    llvm::Value *exn;
    FreeException(llvm::Value *exn) : exn(exn) {}
    void Emit(CodeGenFunction &CGF, Flags flags) {
      CGF.Builder.CreateCall(getFreeExceptionFn(CGF), exn)
        ->setDoesNotThrow();
    }
  };
}

// Emits an exception expression into the given location.  This
// differs from EmitAnyExprToMem only in that, if a final copy-ctor
// call is required, an exception within that copy ctor causes
// std::terminate to be invoked.
static void EmitAnyExprToExn(CodeGenFunction &CGF, const Expr *e,
                             llvm::Value *addr) {
  // Make sure the exception object is cleaned up if there's an
  // exception during initialization.
  CGF.pushFullExprCleanup<FreeException>(EHCleanup, addr);
  EHScopeStack::stable_iterator cleanup = CGF.EHStack.stable_begin();

  // __cxa_allocate_exception returns a void*;  we need to cast this
  // to the appropriate type for the object.
  llvm::Type *ty = CGF.ConvertTypeForMem(e->getType())->getPointerTo();
  llvm::Value *typedAddr = CGF.Builder.CreateBitCast(addr, ty);

  // FIXME: this isn't quite right!  If there's a final unelided call
  // to a copy constructor, then according to [except.terminate]p1 we
  // must call std::terminate() if that constructor throws, because
  // technically that copy occurs after the exception expression is
  // evaluated but before the exception is caught.  But the best way
  // to handle that is to teach EmitAggExpr to do the final copy
  // differently if it can't be elided.
  CGF.EmitAnyExprToMem(e, typedAddr, e->getType().getQualifiers(), 
                       /*IsInit*/ true);

  // Deactivate the cleanup block.
  CGF.DeactivateCleanupBlock(cleanup, cast<llvm::Instruction>(typedAddr));
}

llvm::Value *CodeGenFunction::getExceptionSlot() {
  if (!ExceptionSlot)
    ExceptionSlot = CreateTempAlloca(Int8PtrTy, "exn.slot");
  return ExceptionSlot;
}

llvm::Value *CodeGenFunction::getEHSelectorSlot() {
  if (!EHSelectorSlot)
    EHSelectorSlot = CreateTempAlloca(Int32Ty, "ehselector.slot");
  return EHSelectorSlot;
}

llvm::Value *CodeGenFunction::getExceptionFromSlot() {
  return Builder.CreateLoad(getExceptionSlot(), "exn");
}

llvm::Value *CodeGenFunction::getSelectorFromSlot() {
  return Builder.CreateLoad(getEHSelectorSlot(), "sel");
}

void CodeGenFunction::EmitCXXThrowExpr(const CXXThrowExpr *E) {
  if (!E->getSubExpr()) {
    if (getInvokeDest()) {
      Builder.CreateInvoke(getReThrowFn(*this),
                           getUnreachableBlock(),
                           getInvokeDest())
        ->setDoesNotReturn();
    } else {
      Builder.CreateCall(getReThrowFn(*this))->setDoesNotReturn();
      Builder.CreateUnreachable();
    }

    // throw is an expression, and the expression emitters expect us
    // to leave ourselves at a valid insertion point.
    EmitBlock(createBasicBlock("throw.cont"));

    return;
  }

  QualType ThrowType = E->getSubExpr()->getType();

  // Now allocate the exception object.
  llvm::Type *SizeTy = ConvertType(getContext().getSizeType());
  uint64_t TypeSize = getContext().getTypeSizeInChars(ThrowType).getQuantity();

  llvm::Constant *AllocExceptionFn = getAllocateExceptionFn(*this);
  llvm::CallInst *ExceptionPtr =
    Builder.CreateCall(AllocExceptionFn,
                       llvm::ConstantInt::get(SizeTy, TypeSize),
                       "exception");
  ExceptionPtr->setDoesNotThrow();
  
  EmitAnyExprToExn(*this, E->getSubExpr(), ExceptionPtr);

  // Now throw the exception.
  llvm::Constant *TypeInfo = CGM.GetAddrOfRTTIDescriptor(ThrowType, 
                                                         /*ForEH=*/true);

  // The address of the destructor.  If the exception type has a
  // trivial destructor (or isn't a record), we just pass null.
  llvm::Constant *Dtor = 0;
  if (const RecordType *RecordTy = ThrowType->getAs<RecordType>()) {
    CXXRecordDecl *Record = cast<CXXRecordDecl>(RecordTy->getDecl());
    if (!Record->hasTrivialDestructor()) {
      CXXDestructorDecl *DtorD = Record->getDestructor();
      Dtor = CGM.GetAddrOfCXXDestructor(DtorD, Dtor_Complete);
      Dtor = llvm::ConstantExpr::getBitCast(Dtor, Int8PtrTy);
    }
  }
  if (!Dtor) Dtor = llvm::Constant::getNullValue(Int8PtrTy);

  if (getInvokeDest()) {
    llvm::InvokeInst *ThrowCall =
      Builder.CreateInvoke3(getThrowFn(*this),
                            getUnreachableBlock(), getInvokeDest(),
                            ExceptionPtr, TypeInfo, Dtor);
    ThrowCall->setDoesNotReturn();
  } else {
    llvm::CallInst *ThrowCall =
      Builder.CreateCall3(getThrowFn(*this), ExceptionPtr, TypeInfo, Dtor);
    ThrowCall->setDoesNotReturn();
    Builder.CreateUnreachable();
  }

  // throw is an expression, and the expression emitters expect us
  // to leave ourselves at a valid insertion point.
  EmitBlock(createBasicBlock("throw.cont"));
}

void CodeGenFunction::EmitStartEHSpec(const Decl *D) {
  if (!CGM.getLangOpts().CXXExceptions)
    return;
  
  const FunctionDecl* FD = dyn_cast_or_null<FunctionDecl>(D);
  if (FD == 0)
    return;
  const FunctionProtoType *Proto = FD->getType()->getAs<FunctionProtoType>();
  if (Proto == 0)
    return;

  ExceptionSpecificationType EST = Proto->getExceptionSpecType();
  if (isNoexceptExceptionSpec(EST)) {
    if (Proto->getNoexceptSpec(getContext()) == FunctionProtoType::NR_Nothrow) {
      // noexcept functions are simple terminate scopes.
      EHStack.pushTerminate();
    }
  } else if (EST == EST_Dynamic || EST == EST_DynamicNone) {
    unsigned NumExceptions = Proto->getNumExceptions();
    EHFilterScope *Filter = EHStack.pushFilter(NumExceptions);

    for (unsigned I = 0; I != NumExceptions; ++I) {
      QualType Ty = Proto->getExceptionType(I);
      QualType ExceptType = Ty.getNonReferenceType().getUnqualifiedType();
      llvm::Value *EHType = CGM.GetAddrOfRTTIDescriptor(ExceptType,
                                                        /*ForEH=*/true);
      Filter->setFilter(I, EHType);
    }
  }
}

/// Emit the dispatch block for a filter scope if necessary.
static void emitFilterDispatchBlock(CodeGenFunction &CGF,
                                    EHFilterScope &filterScope) {
  llvm::BasicBlock *dispatchBlock = filterScope.getCachedEHDispatchBlock();
  if (!dispatchBlock) return;
  if (dispatchBlock->use_empty()) {
    delete dispatchBlock;
    return;
  }

  CGF.EmitBlockAfterUses(dispatchBlock);

  // If this isn't a catch-all filter, we need to check whether we got
  // here because the filter triggered.
  if (filterScope.getNumFilters()) {
    // Load the selector value.
    llvm::Value *selector = CGF.getSelectorFromSlot();
    llvm::BasicBlock *unexpectedBB = CGF.createBasicBlock("ehspec.unexpected");

    llvm::Value *zero = CGF.Builder.getInt32(0);
    llvm::Value *failsFilter =
      CGF.Builder.CreateICmpSLT(selector, zero, "ehspec.fails");
    CGF.Builder.CreateCondBr(failsFilter, unexpectedBB, CGF.getEHResumeBlock(false));

    CGF.EmitBlock(unexpectedBB);
  }

  // Call __cxa_call_unexpected.  This doesn't need to be an invoke
  // because __cxa_call_unexpected magically filters exceptions
  // according to the last landing pad the exception was thrown
  // into.  Seriously.
  llvm::Value *exn = CGF.getExceptionFromSlot();
  CGF.Builder.CreateCall(getUnexpectedFn(CGF), exn)
    ->setDoesNotReturn();
  CGF.Builder.CreateUnreachable();
}

void CodeGenFunction::EmitEndEHSpec(const Decl *D) {
  if (!CGM.getLangOpts().CXXExceptions)
    return;
  
  const FunctionDecl* FD = dyn_cast_or_null<FunctionDecl>(D);
  if (FD == 0)
    return;
  const FunctionProtoType *Proto = FD->getType()->getAs<FunctionProtoType>();
  if (Proto == 0)
    return;

  ExceptionSpecificationType EST = Proto->getExceptionSpecType();
  if (isNoexceptExceptionSpec(EST)) {
    if (Proto->getNoexceptSpec(getContext()) == FunctionProtoType::NR_Nothrow) {
      EHStack.popTerminate();
    }
  } else if (EST == EST_Dynamic || EST == EST_DynamicNone) {
    EHFilterScope &filterScope = cast<EHFilterScope>(*EHStack.begin());
    emitFilterDispatchBlock(*this, filterScope);
    EHStack.popFilter();
  }
}

void CodeGenFunction::EmitCXXTryStmt(const CXXTryStmt &S) {
  EnterCXXTryStmt(S);
  EmitStmt(S.getTryBlock());
  ExitCXXTryStmt(S);
}

void CodeGenFunction::EnterCXXTryStmt(const CXXTryStmt &S, bool IsFnTryBlock) {
  unsigned NumHandlers = S.getNumHandlers();
  EHCatchScope *CatchScope = EHStack.pushCatch(NumHandlers);

  for (unsigned I = 0; I != NumHandlers; ++I) {
    const CXXCatchStmt *C = S.getHandler(I);

    llvm::BasicBlock *Handler = createBasicBlock("catch");
    if (C->getExceptionDecl()) {
      // FIXME: Dropping the reference type on the type into makes it
      // impossible to correctly implement catch-by-reference
      // semantics for pointers.  Unfortunately, this is what all
      // existing compilers do, and it's not clear that the standard
      // personality routine is capable of doing this right.  See C++ DR 388:
      //   http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_active.html#388
      QualType CaughtType = C->getCaughtType();
      CaughtType = CaughtType.getNonReferenceType().getUnqualifiedType();

      llvm::Value *TypeInfo = 0;
      if (CaughtType->isObjCObjectPointerType())
        TypeInfo = CGM.getObjCRuntime().GetEHType(CaughtType);
      else
        TypeInfo = CGM.GetAddrOfRTTIDescriptor(CaughtType, /*ForEH=*/true);
      CatchScope->setHandler(I, TypeInfo, Handler);
    } else {
      // No exception decl indicates '...', a catch-all.
      CatchScope->setCatchAllHandler(I, Handler);
    }
  }
}

llvm::BasicBlock *
CodeGenFunction::getEHDispatchBlock(EHScopeStack::stable_iterator si) {
  // The dispatch block for the end of the scope chain is a block that
  // just resumes unwinding.
  if (si == EHStack.stable_end())
    return getEHResumeBlock(true);

  // Otherwise, we should look at the actual scope.
  EHScope &scope = *EHStack.find(si);

  llvm::BasicBlock *dispatchBlock = scope.getCachedEHDispatchBlock();
  if (!dispatchBlock) {
    switch (scope.getKind()) {
    case EHScope::Catch: {
      // Apply a special case to a single catch-all.
      EHCatchScope &catchScope = cast<EHCatchScope>(scope);
      if (catchScope.getNumHandlers() == 1 &&
          catchScope.getHandler(0).isCatchAll()) {
        dispatchBlock = catchScope.getHandler(0).Block;

      // Otherwise, make a dispatch block.
      } else {
        dispatchBlock = createBasicBlock("catch.dispatch");
      }
      break;
    }

    case EHScope::Cleanup:
      dispatchBlock = createBasicBlock("ehcleanup");
      break;

    case EHScope::Filter:
      dispatchBlock = createBasicBlock("filter.dispatch");
      break;

    case EHScope::Terminate:
      dispatchBlock = getTerminateHandler();
      break;
    }
    scope.setCachedEHDispatchBlock(dispatchBlock);
  }
  return dispatchBlock;
}

/// Check whether this is a non-EH scope, i.e. a scope which doesn't
/// affect exception handling.  Currently, the only non-EH scopes are
/// normal-only cleanup scopes.
static bool isNonEHScope(const EHScope &S) {
  switch (S.getKind()) {
  case EHScope::Cleanup:
    return !cast<EHCleanupScope>(S).isEHCleanup();
  case EHScope::Filter:
  case EHScope::Catch:
  case EHScope::Terminate:
    return false;
  }

  llvm_unreachable("Invalid EHScope Kind!");
}

llvm::BasicBlock *CodeGenFunction::getInvokeDestImpl() {
  assert(EHStack.requiresLandingPad());
  assert(!EHStack.empty());

  if (!CGM.getLangOpts().Exceptions)
    return 0;

  // Check the innermost scope for a cached landing pad.  If this is
  // a non-EH cleanup, we'll check enclosing scopes in EmitLandingPad.
  llvm::BasicBlock *LP = EHStack.begin()->getCachedLandingPad();
  if (LP) return LP;

  // Build the landing pad for this scope.
  LP = EmitLandingPad();
  assert(LP);

  // Cache the landing pad on the innermost scope.  If this is a
  // non-EH scope, cache the landing pad on the enclosing scope, too.
  for (EHScopeStack::iterator ir = EHStack.begin(); true; ++ir) {
    ir->setCachedLandingPad(LP);
    if (!isNonEHScope(*ir)) break;
  }

  return LP;
}

// This code contains a hack to work around a design flaw in
// LLVM's EH IR which breaks semantics after inlining.  This same
// hack is implemented in llvm-gcc.
//
// The LLVM EH abstraction is basically a thin veneer over the
// traditional GCC zero-cost design: for each range of instructions
// in the function, there is (at most) one "landing pad" with an
// associated chain of EH actions.  A language-specific personality
// function interprets this chain of actions and (1) decides whether
// or not to resume execution at the landing pad and (2) if so,
// provides an integer indicating why it's stopping.  In LLVM IR,
// the association of a landing pad with a range of instructions is
// achieved via an invoke instruction, the chain of actions becomes
// the arguments to the @llvm.eh.selector call, and the selector
// call returns the integer indicator.  Other than the required
// presence of two intrinsic function calls in the landing pad,
// the IR exactly describes the layout of the output code.
//
// A principal advantage of this design is that it is completely
// language-agnostic; in theory, the LLVM optimizers can treat
// landing pads neutrally, and targets need only know how to lower
// the intrinsics to have a functioning exceptions system (assuming
// that platform exceptions follow something approximately like the
// GCC design).  Unfortunately, landing pads cannot be combined in a
// language-agnostic way: given selectors A and B, there is no way
// to make a single landing pad which faithfully represents the
// semantics of propagating an exception first through A, then
// through B, without knowing how the personality will interpret the
// (lowered form of the) selectors.  This means that inlining has no
// choice but to crudely chain invokes (i.e., to ignore invokes in
// the inlined function, but to turn all unwindable calls into
// invokes), which is only semantically valid if every unwind stops
// at every landing pad.
//
// Therefore, the invoke-inline hack is to guarantee that every
// landing pad has a catch-all.
enum CleanupHackLevel_t {
  /// A level of hack that requires that all landing pads have
  /// catch-alls.
  CHL_MandatoryCatchall,

  /// A level of hack that requires that all landing pads handle
  /// cleanups.
  CHL_MandatoryCleanup,

  /// No hacks at all;  ideal IR generation.
  CHL_Ideal
};
const CleanupHackLevel_t CleanupHackLevel = CHL_MandatoryCleanup;

llvm::BasicBlock *CodeGenFunction::EmitLandingPad() {
  assert(EHStack.requiresLandingPad());

  EHScope &innermostEHScope = *EHStack.find(EHStack.getInnermostEHScope());
  switch (innermostEHScope.getKind()) {
  case EHScope::Terminate:
    return getTerminateLandingPad();

  case EHScope::Catch:
  case EHScope::Cleanup:
  case EHScope::Filter:
    if (llvm::BasicBlock *lpad = innermostEHScope.getCachedLandingPad())
      return lpad;
  }

  // Save the current IR generation state.
  CGBuilderTy::InsertPoint savedIP = Builder.saveAndClearIP();

  const EHPersonality &personality = EHPersonality::get(getLangOpts());

  // Create and configure the landing pad.
  llvm::BasicBlock *lpad = createBasicBlock("lpad");
  EmitBlock(lpad);

  llvm::LandingPadInst *LPadInst =
    Builder.CreateLandingPad(llvm::StructType::get(Int8PtrTy, Int32Ty, NULL),
                             getOpaquePersonalityFn(CGM, personality), 0);

  llvm::Value *LPadExn = Builder.CreateExtractValue(LPadInst, 0);
  Builder.CreateStore(LPadExn, getExceptionSlot());
  llvm::Value *LPadSel = Builder.CreateExtractValue(LPadInst, 1);
  Builder.CreateStore(LPadSel, getEHSelectorSlot());

  // Save the exception pointer.  It's safe to use a single exception
  // pointer per function because EH cleanups can never have nested
  // try/catches.
  // Build the landingpad instruction.

  // Accumulate all the handlers in scope.
  bool hasCatchAll = false;
  bool hasCleanup = false;
  bool hasFilter = false;
  SmallVector<llvm::Value*, 4> filterTypes;
  llvm::SmallPtrSet<llvm::Value*, 4> catchTypes;
  for (EHScopeStack::iterator I = EHStack.begin(), E = EHStack.end();
         I != E; ++I) {

    switch (I->getKind()) {
    case EHScope::Cleanup:
      // If we have a cleanup, remember that.
      hasCleanup = (hasCleanup || cast<EHCleanupScope>(*I).isEHCleanup());
      continue;

    case EHScope::Filter: {
      assert(I.next() == EHStack.end() && "EH filter is not end of EH stack");
      assert(!hasCatchAll && "EH filter reached after catch-all");

      // Filter scopes get added to the landingpad in weird ways.
      EHFilterScope &filter = cast<EHFilterScope>(*I);
      hasFilter = true;

      // Add all the filter values.
      for (unsigned i = 0, e = filter.getNumFilters(); i != e; ++i)
        filterTypes.push_back(filter.getFilter(i));
      goto done;
    }

    case EHScope::Terminate:
      // Terminate scopes are basically catch-alls.
      assert(!hasCatchAll);
      hasCatchAll = true;
      goto done;

    case EHScope::Catch:
      break;
    }

    EHCatchScope &catchScope = cast<EHCatchScope>(*I);
    for (unsigned hi = 0, he = catchScope.getNumHandlers(); hi != he; ++hi) {
      EHCatchScope::Handler handler = catchScope.getHandler(hi);

      // If this is a catch-all, register that and abort.
      if (!handler.Type) {
        assert(!hasCatchAll);
        hasCatchAll = true;
        goto done;
      }

      // Check whether we already have a handler for this type.
      if (catchTypes.insert(handler.Type))
        // If not, add it directly to the landingpad.
        LPadInst->addClause(handler.Type);
    }
  }

 done:
  // If we have a catch-all, add null to the landingpad.
  assert(!(hasCatchAll && hasFilter));
  if (hasCatchAll) {
    LPadInst->addClause(getCatchAllValue(*this));

  // If we have an EH filter, we need to add those handlers in the
  // right place in the landingpad, which is to say, at the end.
  } else if (hasFilter) {
    // Create a filter expression: a constant array indicating which filter
    // types there are. The personality routine only lands here if the filter
    // doesn't match.
    llvm::SmallVector<llvm::Constant*, 8> Filters;
    llvm::ArrayType *AType =
      llvm::ArrayType::get(!filterTypes.empty() ?
                             filterTypes[0]->getType() : Int8PtrTy,
                           filterTypes.size());

    for (unsigned i = 0, e = filterTypes.size(); i != e; ++i)
      Filters.push_back(cast<llvm::Constant>(filterTypes[i]));
    llvm::Constant *FilterArray = llvm::ConstantArray::get(AType, Filters);
    LPadInst->addClause(FilterArray);

    // Also check whether we need a cleanup.
    if (hasCleanup)
      LPadInst->setCleanup(true);

  // Otherwise, signal that we at least have cleanups.
  } else if (CleanupHackLevel == CHL_MandatoryCatchall || hasCleanup) {
    if (CleanupHackLevel == CHL_MandatoryCatchall)
      LPadInst->addClause(getCatchAllValue(*this));
    else
      LPadInst->setCleanup(true);
  }

  assert((LPadInst->getNumClauses() > 0 || LPadInst->isCleanup()) &&
         "landingpad instruction has no clauses!");

  // Tell the backend how to generate the landing pad.
  Builder.CreateBr(getEHDispatchBlock(EHStack.getInnermostEHScope()));

  // Restore the old IR generation state.
  Builder.restoreIP(savedIP);

  return lpad;
}

namespace {
  /// A cleanup to call __cxa_end_catch.  In many cases, the caught
  /// exception type lets us state definitively that the thrown exception
  /// type does not have a destructor.  In particular:
  ///   - Catch-alls tell us nothing, so we have to conservatively
  ///     assume that the thrown exception might have a destructor.
  ///   - Catches by reference behave according to their base types.
  ///   - Catches of non-record types will only trigger for exceptions
  ///     of non-record types, which never have destructors.
  ///   - Catches of record types can trigger for arbitrary subclasses
  ///     of the caught type, so we have to assume the actual thrown
  ///     exception type might have a throwing destructor, even if the
  ///     caught type's destructor is trivial or nothrow.
  struct CallEndCatch : EHScopeStack::Cleanup {
    CallEndCatch(bool MightThrow) : MightThrow(MightThrow) {}
    bool MightThrow;

    void Emit(CodeGenFunction &CGF, Flags flags) {
      if (!MightThrow) {
        CGF.Builder.CreateCall(getEndCatchFn(CGF))->setDoesNotThrow();
        return;
      }

      CGF.EmitCallOrInvoke(getEndCatchFn(CGF));
    }
  };
}

/// Emits a call to __cxa_begin_catch and enters a cleanup to call
/// __cxa_end_catch.
///
/// \param EndMightThrow - true if __cxa_end_catch might throw
static llvm::Value *CallBeginCatch(CodeGenFunction &CGF,
                                   llvm::Value *Exn,
                                   bool EndMightThrow) {
  llvm::CallInst *Call = CGF.Builder.CreateCall(getBeginCatchFn(CGF), Exn);
  Call->setDoesNotThrow();

  CGF.EHStack.pushCleanup<CallEndCatch>(NormalAndEHCleanup, EndMightThrow);

  return Call;
}

/// A "special initializer" callback for initializing a catch
/// parameter during catch initialization.
static void InitCatchParam(CodeGenFunction &CGF,
                           const VarDecl &CatchParam,
                           llvm::Value *ParamAddr) {
  // Load the exception from where the landing pad saved it.
  llvm::Value *Exn = CGF.getExceptionFromSlot();

  CanQualType CatchType =
    CGF.CGM.getContext().getCanonicalType(CatchParam.getType());
  llvm::Type *LLVMCatchTy = CGF.ConvertTypeForMem(CatchType);

  // If we're catching by reference, we can just cast the object
  // pointer to the appropriate pointer.
  if (isa<ReferenceType>(CatchType)) {
    QualType CaughtType = cast<ReferenceType>(CatchType)->getPointeeType();
    bool EndCatchMightThrow = CaughtType->isRecordType();

    // __cxa_begin_catch returns the adjusted object pointer.
    llvm::Value *AdjustedExn = CallBeginCatch(CGF, Exn, EndCatchMightThrow);

    // We have no way to tell the personality function that we're
    // catching by reference, so if we're catching a pointer,
    // __cxa_begin_catch will actually return that pointer by value.
    if (const PointerType *PT = dyn_cast<PointerType>(CaughtType)) {
      QualType PointeeType = PT->getPointeeType();

      // When catching by reference, generally we should just ignore
      // this by-value pointer and use the exception object instead.
      if (!PointeeType->isRecordType()) {

        // Exn points to the struct _Unwind_Exception header, which
        // we have to skip past in order to reach the exception data.
        unsigned HeaderSize =
          CGF.CGM.getTargetCodeGenInfo().getSizeOfUnwindException();
        AdjustedExn = CGF.Builder.CreateConstGEP1_32(Exn, HeaderSize);

      // However, if we're catching a pointer-to-record type that won't
      // work, because the personality function might have adjusted
      // the pointer.  There's actually no way for us to fully satisfy
      // the language/ABI contract here:  we can't use Exn because it
      // might have the wrong adjustment, but we can't use the by-value
      // pointer because it's off by a level of abstraction.
      //
      // The current solution is to dump the adjusted pointer into an
      // alloca, which breaks language semantics (because changing the
      // pointer doesn't change the exception) but at least works.
      // The better solution would be to filter out non-exact matches
      // and rethrow them, but this is tricky because the rethrow
      // really needs to be catchable by other sites at this landing
      // pad.  The best solution is to fix the personality function.
      } else {
        // Pull the pointer for the reference type off.
        llvm::Type *PtrTy =
          cast<llvm::PointerType>(LLVMCatchTy)->getElementType();

        // Create the temporary and write the adjusted pointer into it.
        llvm::Value *ExnPtrTmp = CGF.CreateTempAlloca(PtrTy, "exn.byref.tmp");
        llvm::Value *Casted = CGF.Builder.CreateBitCast(AdjustedExn, PtrTy);
        CGF.Builder.CreateStore(Casted, ExnPtrTmp);

        // Bind the reference to the temporary.
        AdjustedExn = ExnPtrTmp;
      }
    }

    llvm::Value *ExnCast =
      CGF.Builder.CreateBitCast(AdjustedExn, LLVMCatchTy, "exn.byref");
    CGF.Builder.CreateStore(ExnCast, ParamAddr);
    return;
  }

  // Non-aggregates (plus complexes).
  bool IsComplex = false;
  if (!CGF.hasAggregateLLVMType(CatchType) ||
      (IsComplex = CatchType->isAnyComplexType())) {
    llvm::Value *AdjustedExn = CallBeginCatch(CGF, Exn, false);
    
    // If the catch type is a pointer type, __cxa_begin_catch returns
    // the pointer by value.
    if (CatchType->hasPointerRepresentation()) {
      llvm::Value *CastExn =
        CGF.Builder.CreateBitCast(AdjustedExn, LLVMCatchTy, "exn.casted");

      switch (CatchType.getQualifiers().getObjCLifetime()) {
      case Qualifiers::OCL_Strong:
        CastExn = CGF.EmitARCRetainNonBlock(CastExn);
        // fallthrough

      case Qualifiers::OCL_None:
      case Qualifiers::OCL_ExplicitNone:
      case Qualifiers::OCL_Autoreleasing:
        CGF.Builder.CreateStore(CastExn, ParamAddr);
        return;

      case Qualifiers::OCL_Weak:
        CGF.EmitARCInitWeak(ParamAddr, CastExn);
        return;
      }
      llvm_unreachable("bad ownership qualifier!");
    }

    // Otherwise, it returns a pointer into the exception object.

    llvm::Type *PtrTy = LLVMCatchTy->getPointerTo(0); // addrspace 0 ok
    llvm::Value *Cast = CGF.Builder.CreateBitCast(AdjustedExn, PtrTy);

    if (IsComplex) {
      CGF.StoreComplexToAddr(CGF.LoadComplexFromAddr(Cast, /*volatile*/ false),
                             ParamAddr, /*volatile*/ false);
    } else {
      unsigned Alignment =
        CGF.getContext().getDeclAlign(&CatchParam).getQuantity();
      llvm::Value *ExnLoad = CGF.Builder.CreateLoad(Cast, "exn.scalar");
      CGF.EmitStoreOfScalar(ExnLoad, ParamAddr, /*volatile*/ false, Alignment,
                            CatchType);
    }
    return;
  }

  assert(isa<RecordType>(CatchType) && "unexpected catch type!");

  llvm::Type *PtrTy = LLVMCatchTy->getPointerTo(0); // addrspace 0 ok

  // Check for a copy expression.  If we don't have a copy expression,
  // that means a trivial copy is okay.
  const Expr *copyExpr = CatchParam.getInit();
  if (!copyExpr) {
    llvm::Value *rawAdjustedExn = CallBeginCatch(CGF, Exn, true);
    llvm::Value *adjustedExn = CGF.Builder.CreateBitCast(rawAdjustedExn, PtrTy);
    CGF.EmitAggregateCopy(ParamAddr, adjustedExn, CatchType);
    return;
  }

  // We have to call __cxa_get_exception_ptr to get the adjusted
  // pointer before copying.
  llvm::CallInst *rawAdjustedExn =
    CGF.Builder.CreateCall(getGetExceptionPtrFn(CGF), Exn);
  rawAdjustedExn->setDoesNotThrow();

  // Cast that to the appropriate type.
  llvm::Value *adjustedExn = CGF.Builder.CreateBitCast(rawAdjustedExn, PtrTy);

  // The copy expression is defined in terms of an OpaqueValueExpr.
  // Find it and map it to the adjusted expression.
  CodeGenFunction::OpaqueValueMapping
    opaque(CGF, OpaqueValueExpr::findInCopyConstruct(copyExpr),
           CGF.MakeAddrLValue(adjustedExn, CatchParam.getType()));

  // Call the copy ctor in a terminate scope.
  CGF.EHStack.pushTerminate();

  // Perform the copy construction.
  CharUnits Alignment = CGF.getContext().getDeclAlign(&CatchParam);
  CGF.EmitAggExpr(copyExpr,
                  AggValueSlot::forAddr(ParamAddr, Alignment, Qualifiers(),
                                        AggValueSlot::IsNotDestructed,
                                        AggValueSlot::DoesNotNeedGCBarriers,
                                        AggValueSlot::IsNotAliased));

  // Leave the terminate scope.
  CGF.EHStack.popTerminate();

  // Undo the opaque value mapping.
  opaque.pop();

  // Finally we can call __cxa_begin_catch.
  CallBeginCatch(CGF, Exn, true);
}

/// Begins a catch statement by initializing the catch variable and
/// calling __cxa_begin_catch.
static void BeginCatch(CodeGenFunction &CGF, const CXXCatchStmt *S) {
  // We have to be very careful with the ordering of cleanups here:
  //   C++ [except.throw]p4:
  //     The destruction [of the exception temporary] occurs
  //     immediately after the destruction of the object declared in
  //     the exception-declaration in the handler.
  //
  // So the precise ordering is:
  //   1.  Construct catch variable.
  //   2.  __cxa_begin_catch
  //   3.  Enter __cxa_end_catch cleanup
  //   4.  Enter dtor cleanup
  //
  // We do this by using a slightly abnormal initialization process.
  // Delegation sequence:
  //   - ExitCXXTryStmt opens a RunCleanupsScope
  //     - EmitAutoVarAlloca creates the variable and debug info
  //       - InitCatchParam initializes the variable from the exception
  //       - CallBeginCatch calls __cxa_begin_catch
  //       - CallBeginCatch enters the __cxa_end_catch cleanup
  //     - EmitAutoVarCleanups enters the variable destructor cleanup
  //   - EmitCXXTryStmt emits the code for the catch body
  //   - EmitCXXTryStmt close the RunCleanupsScope

  VarDecl *CatchParam = S->getExceptionDecl();
  if (!CatchParam) {
    llvm::Value *Exn = CGF.getExceptionFromSlot();
    CallBeginCatch(CGF, Exn, true);
    return;
  }

  // Emit the local.
  CodeGenFunction::AutoVarEmission var = CGF.EmitAutoVarAlloca(*CatchParam);
  InitCatchParam(CGF, *CatchParam, var.getObjectAddress(CGF));
  CGF.EmitAutoVarCleanups(var);
}

/// Emit the structure of the dispatch block for the given catch scope.
/// It is an invariant that the dispatch block already exists.
static void emitCatchDispatchBlock(CodeGenFunction &CGF,
                                   EHCatchScope &catchScope) {
  llvm::BasicBlock *dispatchBlock = catchScope.getCachedEHDispatchBlock();
  assert(dispatchBlock);

  // If there's only a single catch-all, getEHDispatchBlock returned
  // that catch-all as the dispatch block.
  if (catchScope.getNumHandlers() == 1 &&
      catchScope.getHandler(0).isCatchAll()) {
    assert(dispatchBlock == catchScope.getHandler(0).Block);
    return;
  }

  CGBuilderTy::InsertPoint savedIP = CGF.Builder.saveIP();
  CGF.EmitBlockAfterUses(dispatchBlock);

  // Select the right handler.
  llvm::Value *llvm_eh_typeid_for =
    CGF.CGM.getIntrinsic(llvm::Intrinsic::eh_typeid_for);

  // Load the selector value.
  llvm::Value *selector = CGF.getSelectorFromSlot();

  // Test against each of the exception types we claim to catch.
  for (unsigned i = 0, e = catchScope.getNumHandlers(); ; ++i) {
    assert(i < e && "ran off end of handlers!");
    const EHCatchScope::Handler &handler = catchScope.getHandler(i);

    llvm::Value *typeValue = handler.Type;
    assert(typeValue && "fell into catch-all case!");
    typeValue = CGF.Builder.CreateBitCast(typeValue, CGF.Int8PtrTy);

    // Figure out the next block.
    bool nextIsEnd;
    llvm::BasicBlock *nextBlock;

    // If this is the last handler, we're at the end, and the next
    // block is the block for the enclosing EH scope.
    if (i + 1 == e) {
      nextBlock = CGF.getEHDispatchBlock(catchScope.getEnclosingEHScope());
      nextIsEnd = true;

    // If the next handler is a catch-all, we're at the end, and the
    // next block is that handler.
    } else if (catchScope.getHandler(i+1).isCatchAll()) {
      nextBlock = catchScope.getHandler(i+1).Block;
      nextIsEnd = true;

    // Otherwise, we're not at the end and we need a new block.
    } else {
      nextBlock = CGF.createBasicBlock("catch.fallthrough");
      nextIsEnd = false;
    }

    // Figure out the catch type's index in the LSDA's type table.
    llvm::CallInst *typeIndex =
      CGF.Builder.CreateCall(llvm_eh_typeid_for, typeValue);
    typeIndex->setDoesNotThrow();

    llvm::Value *matchesTypeIndex =
      CGF.Builder.CreateICmpEQ(selector, typeIndex, "matches");
    CGF.Builder.CreateCondBr(matchesTypeIndex, handler.Block, nextBlock);

    // If the next handler is a catch-all, we're completely done.
    if (nextIsEnd) {
      CGF.Builder.restoreIP(savedIP);
      return;
    }
    // Otherwise we need to emit and continue at that block.
    CGF.EmitBlock(nextBlock);
  }
}

void CodeGenFunction::popCatchScope() {
  EHCatchScope &catchScope = cast<EHCatchScope>(*EHStack.begin());
  if (catchScope.hasEHBranches())
    emitCatchDispatchBlock(*this, catchScope);
  EHStack.popCatch();
}

void CodeGenFunction::ExitCXXTryStmt(const CXXTryStmt &S, bool IsFnTryBlock) {
  unsigned NumHandlers = S.getNumHandlers();
  EHCatchScope &CatchScope = cast<EHCatchScope>(*EHStack.begin());
  assert(CatchScope.getNumHandlers() == NumHandlers);

  // If the catch was not required, bail out now.
  if (!CatchScope.hasEHBranches()) {
    EHStack.popCatch();
    return;
  }

  // Emit the structure of the EH dispatch for this catch.
  emitCatchDispatchBlock(*this, CatchScope);

  // Copy the handler blocks off before we pop the EH stack.  Emitting
  // the handlers might scribble on this memory.
  SmallVector<EHCatchScope::Handler, 8> Handlers(NumHandlers);
  memcpy(Handlers.data(), CatchScope.begin(),
         NumHandlers * sizeof(EHCatchScope::Handler));

  EHStack.popCatch();

  // The fall-through block.
  llvm::BasicBlock *ContBB = createBasicBlock("try.cont");

  // We just emitted the body of the try; jump to the continue block.
  if (HaveInsertPoint())
    Builder.CreateBr(ContBB);

  // Determine if we need an implicit rethrow for all these catch handlers;
  // see the comment below.
  bool doImplicitRethrow = false;
  if (IsFnTryBlock)
    doImplicitRethrow = isa<CXXDestructorDecl>(CurCodeDecl) ||
                        isa<CXXConstructorDecl>(CurCodeDecl);

  // Perversely, we emit the handlers backwards precisely because we
  // want them to appear in source order.  In all of these cases, the
  // catch block will have exactly one predecessor, which will be a
  // particular block in the catch dispatch.  However, in the case of
  // a catch-all, one of the dispatch blocks will branch to two
  // different handlers, and EmitBlockAfterUses will cause the second
  // handler to be moved before the first.
  for (unsigned I = NumHandlers; I != 0; --I) {
    llvm::BasicBlock *CatchBlock = Handlers[I-1].Block;
    EmitBlockAfterUses(CatchBlock);

    // Catch the exception if this isn't a catch-all.
    const CXXCatchStmt *C = S.getHandler(I-1);

    // Enter a cleanup scope, including the catch variable and the
    // end-catch.
    RunCleanupsScope CatchScope(*this);

    // Initialize the catch variable and set up the cleanups.
    BeginCatch(*this, C);

    // Perform the body of the catch.
    EmitStmt(C->getHandlerBlock());

    // [except.handle]p11:
    //   The currently handled exception is rethrown if control
    //   reaches the end of a handler of the function-try-block of a
    //   constructor or destructor.

    // It is important that we only do this on fallthrough and not on
    // return.  Note that it's illegal to put a return in a
    // constructor function-try-block's catch handler (p14), so this
    // really only applies to destructors.
    if (doImplicitRethrow && HaveInsertPoint()) {
      EmitCallOrInvoke(getReThrowFn(*this));
      Builder.CreateUnreachable();
      Builder.ClearInsertionPoint();
    }

    // Fall out through the catch cleanups.
    CatchScope.ForceCleanup();

    // Branch out of the try.
    if (HaveInsertPoint())
      Builder.CreateBr(ContBB);
  }

  EmitBlock(ContBB);
}

namespace {
  struct CallEndCatchForFinally : EHScopeStack::Cleanup {
    llvm::Value *ForEHVar;
    llvm::Value *EndCatchFn;
    CallEndCatchForFinally(llvm::Value *ForEHVar, llvm::Value *EndCatchFn)
      : ForEHVar(ForEHVar), EndCatchFn(EndCatchFn) {}

    void Emit(CodeGenFunction &CGF, Flags flags) {
      llvm::BasicBlock *EndCatchBB = CGF.createBasicBlock("finally.endcatch");
      llvm::BasicBlock *CleanupContBB =
        CGF.createBasicBlock("finally.cleanup.cont");

      llvm::Value *ShouldEndCatch =
        CGF.Builder.CreateLoad(ForEHVar, "finally.endcatch");
      CGF.Builder.CreateCondBr(ShouldEndCatch, EndCatchBB, CleanupContBB);
      CGF.EmitBlock(EndCatchBB);
      CGF.EmitCallOrInvoke(EndCatchFn); // catch-all, so might throw
      CGF.EmitBlock(CleanupContBB);
    }
  };

  struct PerformFinally : EHScopeStack::Cleanup {
    const Stmt *Body;
    llvm::Value *ForEHVar;
    llvm::Value *EndCatchFn;
    llvm::Value *RethrowFn;
    llvm::Value *SavedExnVar;

    PerformFinally(const Stmt *Body, llvm::Value *ForEHVar,
                   llvm::Value *EndCatchFn,
                   llvm::Value *RethrowFn, llvm::Value *SavedExnVar)
      : Body(Body), ForEHVar(ForEHVar), EndCatchFn(EndCatchFn),
        RethrowFn(RethrowFn), SavedExnVar(SavedExnVar) {}

    void Emit(CodeGenFunction &CGF, Flags flags) {
      // Enter a cleanup to call the end-catch function if one was provided.
      if (EndCatchFn)
        CGF.EHStack.pushCleanup<CallEndCatchForFinally>(NormalAndEHCleanup,
                                                        ForEHVar, EndCatchFn);

      // Save the current cleanup destination in case there are
      // cleanups in the finally block.
      llvm::Value *SavedCleanupDest =
        CGF.Builder.CreateLoad(CGF.getNormalCleanupDestSlot(),
                               "cleanup.dest.saved");

      // Emit the finally block.
      CGF.EmitStmt(Body);

      // If the end of the finally is reachable, check whether this was
      // for EH.  If so, rethrow.
      if (CGF.HaveInsertPoint()) {
        llvm::BasicBlock *RethrowBB = CGF.createBasicBlock("finally.rethrow");
        llvm::BasicBlock *ContBB = CGF.createBasicBlock("finally.cont");

        llvm::Value *ShouldRethrow =
          CGF.Builder.CreateLoad(ForEHVar, "finally.shouldthrow");
        CGF.Builder.CreateCondBr(ShouldRethrow, RethrowBB, ContBB);

        CGF.EmitBlock(RethrowBB);
        if (SavedExnVar) {
          CGF.EmitCallOrInvoke(RethrowFn, CGF.Builder.CreateLoad(SavedExnVar));
        } else {
          CGF.EmitCallOrInvoke(RethrowFn);
        }
        CGF.Builder.CreateUnreachable();

        CGF.EmitBlock(ContBB);

        // Restore the cleanup destination.
        CGF.Builder.CreateStore(SavedCleanupDest,
                                CGF.getNormalCleanupDestSlot());
      }

      // Leave the end-catch cleanup.  As an optimization, pretend that
      // the fallthrough path was inaccessible; we've dynamically proven
      // that we're not in the EH case along that path.
      if (EndCatchFn) {
        CGBuilderTy::InsertPoint SavedIP = CGF.Builder.saveAndClearIP();
        CGF.PopCleanupBlock();
        CGF.Builder.restoreIP(SavedIP);
      }
    
      // Now make sure we actually have an insertion point or the
      // cleanup gods will hate us.
      CGF.EnsureInsertPoint();
    }
  };
}

/// Enters a finally block for an implementation using zero-cost
/// exceptions.  This is mostly general, but hard-codes some
/// language/ABI-specific behavior in the catch-all sections.
void CodeGenFunction::FinallyInfo::enter(CodeGenFunction &CGF,
                                         const Stmt *body,
                                         llvm::Constant *beginCatchFn,
                                         llvm::Constant *endCatchFn,
                                         llvm::Constant *rethrowFn) {
  assert((beginCatchFn != 0) == (endCatchFn != 0) &&
         "begin/end catch functions not paired");
  assert(rethrowFn && "rethrow function is required");

  BeginCatchFn = beginCatchFn;

  // The rethrow function has one of the following two types:
  //   void (*)()
  //   void (*)(void*)
  // In the latter case we need to pass it the exception object.
  // But we can't use the exception slot because the @finally might
  // have a landing pad (which would overwrite the exception slot).
  llvm::FunctionType *rethrowFnTy =
    cast<llvm::FunctionType>(
      cast<llvm::PointerType>(rethrowFn->getType())->getElementType());
  SavedExnVar = 0;
  if (rethrowFnTy->getNumParams())
    SavedExnVar = CGF.CreateTempAlloca(CGF.Int8PtrTy, "finally.exn");

  // A finally block is a statement which must be executed on any edge
  // out of a given scope.  Unlike a cleanup, the finally block may
  // contain arbitrary control flow leading out of itself.  In
  // addition, finally blocks should always be executed, even if there
  // are no catch handlers higher on the stack.  Therefore, we
  // surround the protected scope with a combination of a normal
  // cleanup (to catch attempts to break out of the block via normal
  // control flow) and an EH catch-all (semantically "outside" any try
  // statement to which the finally block might have been attached).
  // The finally block itself is generated in the context of a cleanup
  // which conditionally leaves the catch-all.

  // Jump destination for performing the finally block on an exception
  // edge.  We'll never actually reach this block, so unreachable is
  // fine.
  RethrowDest = CGF.getJumpDestInCurrentScope(CGF.getUnreachableBlock());

  // Whether the finally block is being executed for EH purposes.
  ForEHVar = CGF.CreateTempAlloca(CGF.Builder.getInt1Ty(), "finally.for-eh");
  CGF.Builder.CreateStore(CGF.Builder.getFalse(), ForEHVar);

  // Enter a normal cleanup which will perform the @finally block.
  CGF.EHStack.pushCleanup<PerformFinally>(NormalCleanup, body,
                                          ForEHVar, endCatchFn,
                                          rethrowFn, SavedExnVar);

  // Enter a catch-all scope.
  llvm::BasicBlock *catchBB = CGF.createBasicBlock("finally.catchall");
  EHCatchScope *catchScope = CGF.EHStack.pushCatch(1);
  catchScope->setCatchAllHandler(0, catchBB);
}

void CodeGenFunction::FinallyInfo::exit(CodeGenFunction &CGF) {
  // Leave the finally catch-all.
  EHCatchScope &catchScope = cast<EHCatchScope>(*CGF.EHStack.begin());
  llvm::BasicBlock *catchBB = catchScope.getHandler(0).Block;

  CGF.popCatchScope();

  // If there are any references to the catch-all block, emit it.
  if (catchBB->use_empty()) {
    delete catchBB;
  } else {
    CGBuilderTy::InsertPoint savedIP = CGF.Builder.saveAndClearIP();
    CGF.EmitBlock(catchBB);

    llvm::Value *exn = 0;

    // If there's a begin-catch function, call it.
    if (BeginCatchFn) {
      exn = CGF.getExceptionFromSlot();
      CGF.Builder.CreateCall(BeginCatchFn, exn)->setDoesNotThrow();
    }

    // If we need to remember the exception pointer to rethrow later, do so.
    if (SavedExnVar) {
      if (!exn) exn = CGF.getExceptionFromSlot();
      CGF.Builder.CreateStore(exn, SavedExnVar);
    }

    // Tell the cleanups in the finally block that we're do this for EH.
    CGF.Builder.CreateStore(CGF.Builder.getTrue(), ForEHVar);

    // Thread a jump through the finally cleanup.
    CGF.EmitBranchThroughCleanup(RethrowDest);

    CGF.Builder.restoreIP(savedIP);
  }

  // Finally, leave the @finally cleanup.
  CGF.PopCleanupBlock();
}

llvm::BasicBlock *CodeGenFunction::getTerminateLandingPad() {
  if (TerminateLandingPad)
    return TerminateLandingPad;

  CGBuilderTy::InsertPoint SavedIP = Builder.saveAndClearIP();

  // This will get inserted at the end of the function.
  TerminateLandingPad = createBasicBlock("terminate.lpad");
  Builder.SetInsertPoint(TerminateLandingPad);

  // Tell the backend that this is a landing pad.
  const EHPersonality &Personality = EHPersonality::get(CGM.getLangOpts());
  llvm::LandingPadInst *LPadInst =
    Builder.CreateLandingPad(llvm::StructType::get(Int8PtrTy, Int32Ty, NULL),
                             getOpaquePersonalityFn(CGM, Personality), 0);
  LPadInst->addClause(getCatchAllValue(*this));

  llvm::CallInst *TerminateCall = Builder.CreateCall(getTerminateFn(*this));
  TerminateCall->setDoesNotReturn();
  TerminateCall->setDoesNotThrow();
  Builder.CreateUnreachable();

  // Restore the saved insertion state.
  Builder.restoreIP(SavedIP);

  return TerminateLandingPad;
}

llvm::BasicBlock *CodeGenFunction::getTerminateHandler() {
  if (TerminateHandler)
    return TerminateHandler;

  CGBuilderTy::InsertPoint SavedIP = Builder.saveAndClearIP();

  // Set up the terminate handler.  This block is inserted at the very
  // end of the function by FinishFunction.
  TerminateHandler = createBasicBlock("terminate.handler");
  Builder.SetInsertPoint(TerminateHandler);
  llvm::CallInst *TerminateCall = Builder.CreateCall(getTerminateFn(*this));
  TerminateCall->setDoesNotReturn();
  TerminateCall->setDoesNotThrow();
  Builder.CreateUnreachable();

  // Restore the saved insertion state.
  Builder.restoreIP(SavedIP);

  return TerminateHandler;
}

llvm::BasicBlock *CodeGenFunction::getEHResumeBlock(bool isCleanup) {
  if (EHResumeBlock) return EHResumeBlock;

  CGBuilderTy::InsertPoint SavedIP = Builder.saveIP();

  // We emit a jump to a notional label at the outermost unwind state.
  EHResumeBlock = createBasicBlock("eh.resume");
  Builder.SetInsertPoint(EHResumeBlock);

  const EHPersonality &Personality = EHPersonality::get(CGM.getLangOpts());

  // This can always be a call because we necessarily didn't find
  // anything on the EH stack which needs our help.
  const char *RethrowName = Personality.CatchallRethrowFn;
  if (RethrowName != 0 && !isCleanup) {
    Builder.CreateCall(getCatchallRethrowFn(*this, RethrowName),
                       getExceptionFromSlot())
      ->setDoesNotReturn();
  } else {
    switch (CleanupHackLevel) {
    case CHL_MandatoryCatchall:
      // In mandatory-catchall mode, we need to use
      // _Unwind_Resume_or_Rethrow, or whatever the personality's
      // equivalent is.
      Builder.CreateCall(getUnwindResumeOrRethrowFn(),
                         getExceptionFromSlot())
        ->setDoesNotReturn();
      break;
    case CHL_MandatoryCleanup: {
      // In mandatory-cleanup mode, we should use 'resume'.

      // Recreate the landingpad's return value for the 'resume' instruction.
      llvm::Value *Exn = getExceptionFromSlot();
      llvm::Value *Sel = getSelectorFromSlot();

      llvm::Type *LPadType = llvm::StructType::get(Exn->getType(),
                                                   Sel->getType(), NULL);
      llvm::Value *LPadVal = llvm::UndefValue::get(LPadType);
      LPadVal = Builder.CreateInsertValue(LPadVal, Exn, 0, "lpad.val");
      LPadVal = Builder.CreateInsertValue(LPadVal, Sel, 1, "lpad.val");

      Builder.CreateResume(LPadVal);
      Builder.restoreIP(SavedIP);
      return EHResumeBlock;
    }
    case CHL_Ideal:
      // In an idealized mode where we don't have to worry about the
      // optimizer combining landing pads, we should just use
      // _Unwind_Resume (or the personality's equivalent).
      Builder.CreateCall(getUnwindResumeFn(), getExceptionFromSlot())
        ->setDoesNotReturn();
      break;
    }
  }

  Builder.CreateUnreachable();

  Builder.restoreIP(SavedIP);

  return EHResumeBlock;
}
