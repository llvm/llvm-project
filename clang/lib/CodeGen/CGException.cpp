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

#include "clang/AST/StmtCXX.h"

#include "llvm/Intrinsics.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Support/CallSite.h"

#include "CGObjCRuntime.h"
#include "CodeGenFunction.h"
#include "CGException.h"
#include "CGCleanup.h"
#include "TargetInfo.h"

using namespace clang;
using namespace CodeGen;

static llvm::Constant *getAllocateExceptionFn(CodeGenFunction &CGF) {
  // void *__cxa_allocate_exception(size_t thrown_size);
  const llvm::Type *SizeTy = CGF.ConvertType(CGF.getContext().getSizeType());
  std::vector<const llvm::Type*> Args(1, SizeTy);

  const llvm::FunctionType *FTy =
  llvm::FunctionType::get(llvm::Type::getInt8PtrTy(CGF.getLLVMContext()),
                          Args, false);

  return CGF.CGM.CreateRuntimeFunction(FTy, "__cxa_allocate_exception");
}

static llvm::Constant *getFreeExceptionFn(CodeGenFunction &CGF) {
  // void __cxa_free_exception(void *thrown_exception);
  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(CGF.getLLVMContext());
  std::vector<const llvm::Type*> Args(1, Int8PtrTy);

  const llvm::FunctionType *FTy =
  llvm::FunctionType::get(llvm::Type::getVoidTy(CGF.getLLVMContext()),
                          Args, false);

  return CGF.CGM.CreateRuntimeFunction(FTy, "__cxa_free_exception");
}

static llvm::Constant *getThrowFn(CodeGenFunction &CGF) {
  // void __cxa_throw(void *thrown_exception, std::type_info *tinfo,
  //                  void (*dest) (void *));

  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(CGF.getLLVMContext());
  std::vector<const llvm::Type*> Args(3, Int8PtrTy);

  const llvm::FunctionType *FTy =
    llvm::FunctionType::get(llvm::Type::getVoidTy(CGF.getLLVMContext()),
                            Args, false);

  return CGF.CGM.CreateRuntimeFunction(FTy, "__cxa_throw");
}

static llvm::Constant *getReThrowFn(CodeGenFunction &CGF) {
  // void __cxa_rethrow();

  const llvm::FunctionType *FTy =
    llvm::FunctionType::get(llvm::Type::getVoidTy(CGF.getLLVMContext()), false);

  return CGF.CGM.CreateRuntimeFunction(FTy, "__cxa_rethrow");
}

static llvm::Constant *getGetExceptionPtrFn(CodeGenFunction &CGF) {
  // void *__cxa_get_exception_ptr(void*);
  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(CGF.getLLVMContext());
  std::vector<const llvm::Type*> Args(1, Int8PtrTy);

  const llvm::FunctionType *FTy =
    llvm::FunctionType::get(Int8PtrTy, Args, false);

  return CGF.CGM.CreateRuntimeFunction(FTy, "__cxa_get_exception_ptr");
}

static llvm::Constant *getBeginCatchFn(CodeGenFunction &CGF) {
  // void *__cxa_begin_catch(void*);

  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(CGF.getLLVMContext());
  std::vector<const llvm::Type*> Args(1, Int8PtrTy);

  const llvm::FunctionType *FTy =
    llvm::FunctionType::get(Int8PtrTy, Args, false);

  return CGF.CGM.CreateRuntimeFunction(FTy, "__cxa_begin_catch");
}

static llvm::Constant *getEndCatchFn(CodeGenFunction &CGF) {
  // void __cxa_end_catch();

  const llvm::FunctionType *FTy =
    llvm::FunctionType::get(llvm::Type::getVoidTy(CGF.getLLVMContext()), false);

  return CGF.CGM.CreateRuntimeFunction(FTy, "__cxa_end_catch");
}

static llvm::Constant *getUnexpectedFn(CodeGenFunction &CGF) {
  // void __cxa_call_unexepcted(void *thrown_exception);

  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(CGF.getLLVMContext());
  std::vector<const llvm::Type*> Args(1, Int8PtrTy);

  const llvm::FunctionType *FTy =
    llvm::FunctionType::get(llvm::Type::getVoidTy(CGF.getLLVMContext()),
                            Args, false);

  return CGF.CGM.CreateRuntimeFunction(FTy, "__cxa_call_unexpected");
}

llvm::Constant *CodeGenFunction::getUnwindResumeOrRethrowFn() {
  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(getLLVMContext());
  std::vector<const llvm::Type*> Args(1, Int8PtrTy);

  const llvm::FunctionType *FTy =
    llvm::FunctionType::get(llvm::Type::getVoidTy(getLLVMContext()), Args,
                            false);

  if (CGM.getLangOptions().SjLjExceptions)
    return CGM.CreateRuntimeFunction(FTy, "_Unwind_SjLj_Resume_or_Rethrow");
  return CGM.CreateRuntimeFunction(FTy, "_Unwind_Resume_or_Rethrow");
}

static llvm::Constant *getTerminateFn(CodeGenFunction &CGF) {
  // void __terminate();

  const llvm::FunctionType *FTy =
    llvm::FunctionType::get(llvm::Type::getVoidTy(CGF.getLLVMContext()), false);

  return CGF.CGM.CreateRuntimeFunction(FTy, 
      CGF.CGM.getLangOptions().CPlusPlus ? "_ZSt9terminatev" : "abort");
}

static llvm::Constant *getCatchallRethrowFn(CodeGenFunction &CGF,
                                            llvm::StringRef Name) {
  const llvm::Type *Int8PtrTy =
    llvm::Type::getInt8PtrTy(CGF.getLLVMContext());
  std::vector<const llvm::Type*> Args(1, Int8PtrTy);

  const llvm::Type *VoidTy = llvm::Type::getVoidTy(CGF.getLLVMContext());
  const llvm::FunctionType *FTy = llvm::FunctionType::get(VoidTy, Args, false);

  return CGF.CGM.CreateRuntimeFunction(FTy, Name);
}

const EHPersonality EHPersonality::GNU_C("__gcc_personality_v0");
const EHPersonality EHPersonality::GNU_C_SJLJ("__gcc_personality_sj0");
const EHPersonality EHPersonality::NeXT_ObjC("__objc_personality_v0");
const EHPersonality EHPersonality::GNU_CPlusPlus("__gxx_personality_v0");
const EHPersonality EHPersonality::GNU_CPlusPlus_SJLJ("__gxx_personality_sj0");
const EHPersonality EHPersonality::GNU_ObjC("__gnu_objc_personality_v0",
                                            "objc_exception_throw");
const EHPersonality EHPersonality::GNU_ObjCXX("__gnustep_objcxx_personality_v0");

static const EHPersonality &getCPersonality(const LangOptions &L) {
  if (L.SjLjExceptions)
    return EHPersonality::GNU_C_SJLJ;
  return EHPersonality::GNU_C;
}

static const EHPersonality &getObjCPersonality(const LangOptions &L) {
  if (L.NeXTRuntime) {
    if (L.ObjCNonFragileABI) return EHPersonality::NeXT_ObjC;
    else return getCPersonality(L);
  } else {
    return EHPersonality::GNU_ObjC;
  }
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
  // The ObjC personality defers to the C++ personality for non-ObjC
  // handlers.  Unlike the C++ case, we use the same personality
  // function on targets using (backend-driven) SJLJ EH.
  if (L.NeXTRuntime) {
    if (L.ObjCNonFragileABI)
      return EHPersonality::NeXT_ObjC;

    // In the fragile ABI, just use C++ exception handling and hope
    // they're not doing crazy exception mixing.
    else
      return getCXXPersonality(L);
  }

  // The GNU runtime's personality function inherently doesn't support
  // mixed EH.  Use the C++ personality just to avoid returning null.
  return EHPersonality::GNU_ObjCXX;
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
    CGM.CreateRuntimeFunction(llvm::FunctionType::get(
                                llvm::Type::getInt32Ty(CGM.getLLVMContext()),
                                true),
                              Personality.getPersonalityFnName());
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

    // Otherwise, it has to be a selector call.
    if (!isa<llvm::EHSelectorInst>(User)) return false;

    llvm::EHSelectorInst *Selector = cast<llvm::EHSelectorInst>(User);
    for (unsigned I = 2, E = Selector->getNumArgOperands(); I != E; ++I) {
      // Look for something that would've been returned by the ObjC
      // runtime's GetEHType() method.
      llvm::GlobalVariable *GV
        = dyn_cast<llvm::GlobalVariable>(Selector->getArgOperand(I));
      if (!GV) continue;

      // ObjC EH selector entries are always global variables with
      // names starting like this.
      if (GV->getName().startswith("OBJC_EHTYPE"))
        return false;
    }
  }

  return true;
}

/// Try to use the C++ personality function in ObjC++.  Not doing this
/// can cause some incompatibilities with gcc, which is more
/// aggressive about only using the ObjC++ personality in a function
/// when it really needs it.
void CodeGenModule::SimplifyPersonality() {
  // For now, this is really a Darwin-specific operation.
  if (Context.Target.getTriple().getOS() != llvm::Triple::Darwin)
    return;

  // If we're not in ObjC++ -fexceptions, there's nothing to do.
  if (!Features.CPlusPlus || !Features.ObjC1 || !Features.Exceptions)
    return;

  const EHPersonality &ObjCXX = EHPersonality::get(Features);
  const EHPersonality &CXX = getCXXPersonality(Features);
  if (&ObjCXX == &CXX ||
      ObjCXX.getPersonalityFnName() == CXX.getPersonalityFnName())
    return;

  llvm::Function *Fn =
    getModule().getFunction(ObjCXX.getPersonalityFnName());

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

/// Returns the value to inject into a selector to indicate the
/// presence of a cleanup.
static llvm::Constant *getCleanupValue(CodeGenFunction &CGF) {
  return llvm::ConstantInt::get(CGF.Builder.getInt32Ty(), 0);
}

namespace {
  /// A cleanup to free the exception object if its initialization
  /// throws.
  struct FreeException {
    static void Emit(CodeGenFunction &CGF, bool forEH,
                     llvm::Value *exn) {
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
  const llvm::Type *ty = CGF.ConvertTypeForMem(e->getType())->getPointerTo();
  llvm::Value *typedAddr = CGF.Builder.CreateBitCast(addr, ty);

  // FIXME: this isn't quite right!  If there's a final unelided call
  // to a copy constructor, then according to [except.terminate]p1 we
  // must call std::terminate() if that constructor throws, because
  // technically that copy occurs after the exception expression is
  // evaluated but before the exception is caught.  But the best way
  // to handle that is to teach EmitAggExpr to do the final copy
  // differently if it can't be elided.
  CGF.EmitAnyExprToMem(e, typedAddr, /*Volatile*/ false, /*IsInit*/ true);

  // Deactivate the cleanup block.
  CGF.DeactivateCleanupBlock(cleanup);
}

llvm::Value *CodeGenFunction::getExceptionSlot() {
  if (!ExceptionSlot) {
    const llvm::Type *i8p = llvm::Type::getInt8PtrTy(getLLVMContext());
    ExceptionSlot = CreateTempAlloca(i8p, "exn.slot");
  }
  return ExceptionSlot;
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
  const llvm::Type *SizeTy = ConvertType(getContext().getSizeType());
  uint64_t TypeSize = getContext().getTypeSizeInChars(ThrowType).getQuantity();

  llvm::Constant *AllocExceptionFn = getAllocateExceptionFn(*this);
  llvm::CallInst *ExceptionPtr =
    Builder.CreateCall(AllocExceptionFn,
                       llvm::ConstantInt::get(SizeTy, TypeSize),
                       "exception");
  ExceptionPtr->setDoesNotThrow();
  
  EmitAnyExprToExn(*this, E->getSubExpr(), ExceptionPtr);

  // Now throw the exception.
  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(getLLVMContext());
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
  if (!CGM.getLangOptions().CXXExceptions)
    return;
  
  const FunctionDecl* FD = dyn_cast_or_null<FunctionDecl>(D);
  if (FD == 0)
    return;
  const FunctionProtoType *Proto = FD->getType()->getAs<FunctionProtoType>();
  if (Proto == 0)
    return;

  assert(!Proto->hasAnyExceptionSpec() && "function with parameter pack");

  if (!Proto->hasExceptionSpec())
    return;

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

void CodeGenFunction::EmitEndEHSpec(const Decl *D) {
  if (!CGM.getLangOptions().CXXExceptions)
    return;
  
  const FunctionDecl* FD = dyn_cast_or_null<FunctionDecl>(D);
  if (FD == 0)
    return;
  const FunctionProtoType *Proto = FD->getType()->getAs<FunctionProtoType>();
  if (Proto == 0)
    return;

  if (!Proto->hasExceptionSpec())
    return;

  EHStack.popFilter();
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

  // Suppress warning.
  return false;
}

llvm::BasicBlock *CodeGenFunction::getInvokeDestImpl() {
  assert(EHStack.requiresLandingPad());
  assert(!EHStack.empty());

  if (!CGM.getLangOptions().Exceptions)
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

llvm::BasicBlock *CodeGenFunction::EmitLandingPad() {
  assert(EHStack.requiresLandingPad());

  // This function contains a hack to work around a design flaw in
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
  const bool UseInvokeInlineHack = true;

  for (EHScopeStack::iterator ir = EHStack.begin(); ; ) {
    assert(ir != EHStack.end() &&
           "stack requiring landing pad is nothing but non-EH scopes?");

    // If this is a terminate scope, just use the singleton terminate
    // landing pad.
    if (isa<EHTerminateScope>(*ir))
      return getTerminateLandingPad();

    // If this isn't an EH scope, iterate; otherwise break out.
    if (!isNonEHScope(*ir)) break;
    ++ir;

    // We haven't checked this scope for a cached landing pad yet.
    if (llvm::BasicBlock *LP = ir->getCachedLandingPad())
      return LP;
  }

  // Save the current IR generation state.
  CGBuilderTy::InsertPoint SavedIP = Builder.saveAndClearIP();

  const EHPersonality &Personality = EHPersonality::get(getLangOptions());

  // Create and configure the landing pad.
  llvm::BasicBlock *LP = createBasicBlock("lpad");
  EmitBlock(LP);

  // Save the exception pointer.  It's safe to use a single exception
  // pointer per function because EH cleanups can never have nested
  // try/catches.
  llvm::CallInst *Exn =
    Builder.CreateCall(CGM.getIntrinsic(llvm::Intrinsic::eh_exception), "exn");
  Exn->setDoesNotThrow();
  Builder.CreateStore(Exn, getExceptionSlot());
  
  // Build the selector arguments.
  llvm::SmallVector<llvm::Value*, 8> EHSelector;
  EHSelector.push_back(Exn);
  EHSelector.push_back(getOpaquePersonalityFn(CGM, Personality));

  // Accumulate all the handlers in scope.
  llvm::DenseMap<llvm::Value*, UnwindDest> EHHandlers;
  UnwindDest CatchAll;
  bool HasEHCleanup = false;
  bool HasEHFilter = false;
  llvm::SmallVector<llvm::Value*, 8> EHFilters;
  for (EHScopeStack::iterator I = EHStack.begin(), E = EHStack.end();
         I != E; ++I) {

    switch (I->getKind()) {
    case EHScope::Cleanup:
      if (!HasEHCleanup)
        HasEHCleanup = cast<EHCleanupScope>(*I).isEHCleanup();
      // We otherwise don't care about cleanups.
      continue;

    case EHScope::Filter: {
      assert(I.next() == EHStack.end() && "EH filter is not end of EH stack");
      assert(!CatchAll.isValid() && "EH filter reached after catch-all");

      // Filter scopes get added to the selector in wierd ways.
      EHFilterScope &Filter = cast<EHFilterScope>(*I);
      HasEHFilter = true;

      // Add all the filter values which we aren't already explicitly
      // catching.
      for (unsigned I = 0, E = Filter.getNumFilters(); I != E; ++I) {
        llvm::Value *FV = Filter.getFilter(I);
        if (!EHHandlers.count(FV))
          EHFilters.push_back(FV);
      }
      goto done;
    }

    case EHScope::Terminate:
      // Terminate scopes are basically catch-alls.
      assert(!CatchAll.isValid());
      CatchAll = UnwindDest(getTerminateHandler(),
                            EHStack.getEnclosingEHCleanup(I),
                            cast<EHTerminateScope>(*I).getDestIndex());
      goto done;

    case EHScope::Catch:
      break;
    }

    EHCatchScope &Catch = cast<EHCatchScope>(*I);
    for (unsigned HI = 0, HE = Catch.getNumHandlers(); HI != HE; ++HI) {
      EHCatchScope::Handler Handler = Catch.getHandler(HI);

      // Catch-all.  We should only have one of these per catch.
      if (!Handler.Type) {
        assert(!CatchAll.isValid());
        CatchAll = UnwindDest(Handler.Block,
                              EHStack.getEnclosingEHCleanup(I),
                              Handler.Index);
        continue;
      }

      // Check whether we already have a handler for this type.
      UnwindDest &Dest = EHHandlers[Handler.Type];
      if (Dest.isValid()) continue;

      EHSelector.push_back(Handler.Type);
      Dest = UnwindDest(Handler.Block,
                        EHStack.getEnclosingEHCleanup(I),
                        Handler.Index);
    }

    // Stop if we found a catch-all.
    if (CatchAll.isValid()) break;
  }

 done:
  unsigned LastToEmitInLoop = EHSelector.size();

  // If we have a catch-all, add null to the selector.
  if (CatchAll.isValid()) {
    EHSelector.push_back(getCatchAllValue(*this));

  // If we have an EH filter, we need to add those handlers in the
  // right place in the selector, which is to say, at the end.
  } else if (HasEHFilter) {
    // Create a filter expression: an integer constant saying how many
    // filters there are (+1 to avoid ambiguity with 0 for cleanup),
    // followed by the filter types.  The personality routine only
    // lands here if the filter doesn't match.
    EHSelector.push_back(llvm::ConstantInt::get(Builder.getInt32Ty(),
                                                EHFilters.size() + 1));
    EHSelector.append(EHFilters.begin(), EHFilters.end());

    // Also check whether we need a cleanup.
    if (UseInvokeInlineHack || HasEHCleanup)
      EHSelector.push_back(UseInvokeInlineHack
                           ? getCatchAllValue(*this)
                           : getCleanupValue(*this));

  // Otherwise, signal that we at least have cleanups.
  } else if (UseInvokeInlineHack || HasEHCleanup) {
    EHSelector.push_back(UseInvokeInlineHack
                         ? getCatchAllValue(*this)
                         : getCleanupValue(*this));
  } else {
    assert(LastToEmitInLoop > 2);
    LastToEmitInLoop--;
  }

  assert(EHSelector.size() >= 3 && "selector call has only two arguments!");

  // Tell the backend how to generate the landing pad.
  llvm::CallInst *Selection =
    Builder.CreateCall(CGM.getIntrinsic(llvm::Intrinsic::eh_selector),
                       EHSelector.begin(), EHSelector.end(), "eh.selector");
  Selection->setDoesNotThrow();
  
  // Select the right handler.
  llvm::Value *llvm_eh_typeid_for =
    CGM.getIntrinsic(llvm::Intrinsic::eh_typeid_for);

  // The results of llvm_eh_typeid_for aren't reliable --- at least
  // not locally --- so we basically have to do this as an 'if' chain.
  // We walk through the first N-1 catch clauses, testing and chaining,
  // and then fall into the final clause (which is either a cleanup, a
  // filter (possibly with a cleanup), a catch-all, or another catch).
  for (unsigned I = 2; I != LastToEmitInLoop; ++I) {
    llvm::Value *Type = EHSelector[I];
    UnwindDest Dest = EHHandlers[Type];
    assert(Dest.isValid() && "no handler entry for value in selector?");

    // Figure out where to branch on a match.  As a debug code-size
    // optimization, if the scope depth matches the innermost cleanup,
    // we branch directly to the catch handler.
    llvm::BasicBlock *Match = Dest.getBlock();
    bool MatchNeedsCleanup =
      Dest.getScopeDepth() != EHStack.getInnermostEHCleanup();
    if (MatchNeedsCleanup)
      Match = createBasicBlock("eh.match");

    llvm::BasicBlock *Next = createBasicBlock("eh.next");

    // Check whether the exception matches.
    llvm::CallInst *Id
      = Builder.CreateCall(llvm_eh_typeid_for,
                           Builder.CreateBitCast(Type, Int8PtrTy));
    Id->setDoesNotThrow();
    Builder.CreateCondBr(Builder.CreateICmpEQ(Selection, Id),
                         Match, Next);
    
    // Emit match code if necessary.
    if (MatchNeedsCleanup) {
      EmitBlock(Match);
      EmitBranchThroughEHCleanup(Dest);
    }

    // Continue to the next match.
    EmitBlock(Next);
  }

  // Emit the final case in the selector.
  // This might be a catch-all....
  if (CatchAll.isValid()) {
    assert(isa<llvm::ConstantPointerNull>(EHSelector.back()));
    EmitBranchThroughEHCleanup(CatchAll);

  // ...or an EH filter...
  } else if (HasEHFilter) {
    llvm::Value *SavedSelection = Selection;

    // First, unwind out to the outermost scope if necessary.
    if (EHStack.hasEHCleanups()) {
      // The end here might not dominate the beginning, so we might need to
      // save the selector if we need it.
      llvm::AllocaInst *SelectorVar = 0;
      if (HasEHCleanup) {
        SelectorVar = CreateTempAlloca(Builder.getInt32Ty(), "selector.var");
        Builder.CreateStore(Selection, SelectorVar);
      }

      llvm::BasicBlock *CleanupContBB = createBasicBlock("ehspec.cleanup.cont");
      EmitBranchThroughEHCleanup(UnwindDest(CleanupContBB, EHStack.stable_end(),
                                            EHStack.getNextEHDestIndex()));
      EmitBlock(CleanupContBB);

      if (HasEHCleanup)
        SavedSelection = Builder.CreateLoad(SelectorVar, "ehspec.saved-selector");
    }

    // If there was a cleanup, we'll need to actually check whether we
    // landed here because the filter triggered.
    if (UseInvokeInlineHack || HasEHCleanup) {
      llvm::BasicBlock *RethrowBB = createBasicBlock("cleanup");
      llvm::BasicBlock *UnexpectedBB = createBasicBlock("ehspec.unexpected");

      llvm::Constant *Zero = llvm::ConstantInt::get(Builder.getInt32Ty(), 0);
      llvm::Value *FailsFilter =
        Builder.CreateICmpSLT(SavedSelection, Zero, "ehspec.fails");
      Builder.CreateCondBr(FailsFilter, UnexpectedBB, RethrowBB);

      // The rethrow block is where we land if this was a cleanup.
      // TODO: can this be _Unwind_Resume if the InvokeInlineHack is off?
      EmitBlock(RethrowBB);
      Builder.CreateCall(getUnwindResumeOrRethrowFn(),
                         Builder.CreateLoad(getExceptionSlot()))
        ->setDoesNotReturn();
      Builder.CreateUnreachable();

      EmitBlock(UnexpectedBB);
    }

    // Call __cxa_call_unexpected.  This doesn't need to be an invoke
    // because __cxa_call_unexpected magically filters exceptions
    // according to the last landing pad the exception was thrown
    // into.  Seriously.
    Builder.CreateCall(getUnexpectedFn(*this),
                       Builder.CreateLoad(getExceptionSlot()))
      ->setDoesNotReturn();
    Builder.CreateUnreachable();

  // ...or a normal catch handler...
  } else if (!UseInvokeInlineHack && !HasEHCleanup) {
    llvm::Value *Type = EHSelector.back();
    EmitBranchThroughEHCleanup(EHHandlers[Type]);

  // ...or a cleanup.
  } else {
    EmitBranchThroughEHCleanup(getRethrowDest());
  }

  // Restore the old IR generation state.
  Builder.restoreIP(SavedIP);

  return LP;
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

    void Emit(CodeGenFunction &CGF, bool IsForEH) {
      if (!MightThrow) {
        CGF.Builder.CreateCall(getEndCatchFn(CGF))->setDoesNotThrow();
        return;
      }

      CGF.EmitCallOrInvoke(getEndCatchFn(CGF), 0, 0);
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
  llvm::Value *Exn = CGF.Builder.CreateLoad(CGF.getExceptionSlot(), "exn");

  CanQualType CatchType =
    CGF.CGM.getContext().getCanonicalType(CatchParam.getType());
  const llvm::Type *LLVMCatchTy = CGF.ConvertTypeForMem(CatchType);

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
        const llvm::Type *PtrTy =
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
      CGF.Builder.CreateStore(CastExn, ParamAddr);
      return;
    }

    // Otherwise, it returns a pointer into the exception object.

    const llvm::Type *PtrTy = LLVMCatchTy->getPointerTo(0); // addrspace 0 ok
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

  const llvm::Type *PtrTy = LLVMCatchTy->getPointerTo(0); // addrspace 0 ok

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
  CGF.EmitAggExpr(copyExpr, AggValueSlot::forAddr(ParamAddr, false, false));

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
    llvm::Value *Exn = CGF.Builder.CreateLoad(CGF.getExceptionSlot(), "exn");
    CallBeginCatch(CGF, Exn, true);
    return;
  }

  // Emit the local.
  CodeGenFunction::AutoVarEmission var = CGF.EmitAutoVarAlloca(*CatchParam);
  InitCatchParam(CGF, *CatchParam, var.getObjectAddress(CGF));
  CGF.EmitAutoVarCleanups(var);
}

namespace {
  struct CallRethrow : EHScopeStack::Cleanup {
    void Emit(CodeGenFunction &CGF, bool IsForEH) {
      CGF.EmitCallOrInvoke(getReThrowFn(CGF), 0, 0);
    }
  };
}

void CodeGenFunction::ExitCXXTryStmt(const CXXTryStmt &S, bool IsFnTryBlock) {
  unsigned NumHandlers = S.getNumHandlers();
  EHCatchScope &CatchScope = cast<EHCatchScope>(*EHStack.begin());
  assert(CatchScope.getNumHandlers() == NumHandlers);

  // Copy the handler blocks off before we pop the EH stack.  Emitting
  // the handlers might scribble on this memory.
  llvm::SmallVector<EHCatchScope::Handler, 8> Handlers(NumHandlers);
  memcpy(Handlers.data(), CatchScope.begin(),
         NumHandlers * sizeof(EHCatchScope::Handler));
  EHStack.popCatch();

  // The fall-through block.
  llvm::BasicBlock *ContBB = createBasicBlock("try.cont");

  // We just emitted the body of the try; jump to the continue block.
  if (HaveInsertPoint())
    Builder.CreateBr(ContBB);

  // Determine if we need an implicit rethrow for all these catch handlers.
  bool ImplicitRethrow = false;
  if (IsFnTryBlock)
    ImplicitRethrow = isa<CXXDestructorDecl>(CurCodeDecl) ||
                      isa<CXXConstructorDecl>(CurCodeDecl);

  for (unsigned I = 0; I != NumHandlers; ++I) {
    llvm::BasicBlock *CatchBlock = Handlers[I].Block;
    EmitBlock(CatchBlock);

    // Catch the exception if this isn't a catch-all.
    const CXXCatchStmt *C = S.getHandler(I);

    // Enter a cleanup scope, including the catch variable and the
    // end-catch.
    RunCleanupsScope CatchScope(*this);

    // Initialize the catch variable and set up the cleanups.
    BeginCatch(*this, C);

    // If there's an implicit rethrow, push a normal "cleanup" to call
    // _cxa_rethrow.  This needs to happen before __cxa_end_catch is
    // called, and so it is pushed after BeginCatch.
    if (ImplicitRethrow)
      EHStack.pushCleanup<CallRethrow>(NormalCleanup);

    // Perform the body of the catch.
    EmitStmt(C->getHandlerBlock());

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

    void Emit(CodeGenFunction &CGF, bool IsForEH) {
      llvm::BasicBlock *EndCatchBB = CGF.createBasicBlock("finally.endcatch");
      llvm::BasicBlock *CleanupContBB =
        CGF.createBasicBlock("finally.cleanup.cont");

      llvm::Value *ShouldEndCatch =
        CGF.Builder.CreateLoad(ForEHVar, "finally.endcatch");
      CGF.Builder.CreateCondBr(ShouldEndCatch, EndCatchBB, CleanupContBB);
      CGF.EmitBlock(EndCatchBB);
      CGF.EmitCallOrInvoke(EndCatchFn, 0, 0); // catch-all, so might throw
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

    void Emit(CodeGenFunction &CGF, bool IsForEH) {
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
          llvm::Value *Args[] = { CGF.Builder.CreateLoad(SavedExnVar) };
          CGF.EmitCallOrInvoke(RethrowFn, Args, Args+1);
        } else {
          CGF.EmitCallOrInvoke(RethrowFn, 0, 0);
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
CodeGenFunction::FinallyInfo
CodeGenFunction::EnterFinallyBlock(const Stmt *Body,
                                   llvm::Constant *BeginCatchFn,
                                   llvm::Constant *EndCatchFn,
                                   llvm::Constant *RethrowFn) {
  assert((BeginCatchFn != 0) == (EndCatchFn != 0) &&
         "begin/end catch functions not paired");
  assert(RethrowFn && "rethrow function is required");

  // The rethrow function has one of the following two types:
  //   void (*)()
  //   void (*)(void*)
  // In the latter case we need to pass it the exception object.
  // But we can't use the exception slot because the @finally might
  // have a landing pad (which would overwrite the exception slot).
  const llvm::FunctionType *RethrowFnTy =
    cast<llvm::FunctionType>(
      cast<llvm::PointerType>(RethrowFn->getType())
      ->getElementType());
  llvm::Value *SavedExnVar = 0;
  if (RethrowFnTy->getNumParams())
    SavedExnVar = CreateTempAlloca(Builder.getInt8PtrTy(), "finally.exn");

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

  FinallyInfo Info;

  // Jump destination for performing the finally block on an exception
  // edge.  We'll never actually reach this block, so unreachable is
  // fine.
  JumpDest RethrowDest = getJumpDestInCurrentScope(getUnreachableBlock());

  // Whether the finally block is being executed for EH purposes.
  llvm::AllocaInst *ForEHVar = CreateTempAlloca(Builder.getInt1Ty(),
                                                "finally.for-eh");
  InitTempAlloca(ForEHVar, llvm::ConstantInt::getFalse(getLLVMContext()));

  // Enter a normal cleanup which will perform the @finally block.
  EHStack.pushCleanup<PerformFinally>(NormalCleanup, Body,
                                      ForEHVar, EndCatchFn,
                                      RethrowFn, SavedExnVar);

  // Enter a catch-all scope.
  llvm::BasicBlock *CatchAllBB = createBasicBlock("finally.catchall");
  CGBuilderTy::InsertPoint SavedIP = Builder.saveIP();
  Builder.SetInsertPoint(CatchAllBB);

  // If there's a begin-catch function, call it.
  if (BeginCatchFn) {
    Builder.CreateCall(BeginCatchFn, Builder.CreateLoad(getExceptionSlot()))
      ->setDoesNotThrow();
  }

  // If we need to remember the exception pointer to rethrow later, do so.
  if (SavedExnVar) {
    llvm::Value *SavedExn = Builder.CreateLoad(getExceptionSlot());
    Builder.CreateStore(SavedExn, SavedExnVar);
  }

  // Tell the finally block that we're in EH.
  Builder.CreateStore(llvm::ConstantInt::getTrue(getLLVMContext()), ForEHVar);

  // Thread a jump through the finally cleanup.
  EmitBranchThroughCleanup(RethrowDest);

  Builder.restoreIP(SavedIP);

  EHCatchScope *CatchScope = EHStack.pushCatch(1);
  CatchScope->setCatchAllHandler(0, CatchAllBB);

  return Info;
}

void CodeGenFunction::ExitFinallyBlock(FinallyInfo &Info) {
  // Leave the finally catch-all.
  EHCatchScope &Catch = cast<EHCatchScope>(*EHStack.begin());
  llvm::BasicBlock *CatchAllBB = Catch.getHandler(0).Block;
  EHStack.popCatch();

  // And leave the normal cleanup.
  PopCleanupBlock();

  CGBuilderTy::InsertPoint SavedIP = Builder.saveAndClearIP();
  EmitBlock(CatchAllBB, true);

  Builder.restoreIP(SavedIP);
}

llvm::BasicBlock *CodeGenFunction::getTerminateLandingPad() {
  if (TerminateLandingPad)
    return TerminateLandingPad;

  CGBuilderTy::InsertPoint SavedIP = Builder.saveAndClearIP();

  // This will get inserted at the end of the function.
  TerminateLandingPad = createBasicBlock("terminate.lpad");
  Builder.SetInsertPoint(TerminateLandingPad);

  // Tell the backend that this is a landing pad.
  llvm::CallInst *Exn =
    Builder.CreateCall(CGM.getIntrinsic(llvm::Intrinsic::eh_exception), "exn");
  Exn->setDoesNotThrow();

  const EHPersonality &Personality = EHPersonality::get(CGM.getLangOptions());
  
  // Tell the backend what the exception table should be:
  // nothing but a catch-all.
  llvm::Value *Args[3] = { Exn, getOpaquePersonalityFn(CGM, Personality),
                           getCatchAllValue(*this) };
  Builder.CreateCall(CGM.getIntrinsic(llvm::Intrinsic::eh_selector),
                     Args, Args+3, "eh.selector")
    ->setDoesNotThrow();

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

CodeGenFunction::UnwindDest CodeGenFunction::getRethrowDest() {
  if (RethrowBlock.isValid()) return RethrowBlock;

  CGBuilderTy::InsertPoint SavedIP = Builder.saveIP();

  // We emit a jump to a notional label at the outermost unwind state.
  llvm::BasicBlock *Unwind = createBasicBlock("eh.resume");
  Builder.SetInsertPoint(Unwind);

  const EHPersonality &Personality = EHPersonality::get(CGM.getLangOptions());

  // This can always be a call because we necessarily didn't find
  // anything on the EH stack which needs our help.
  llvm::StringRef RethrowName = Personality.getCatchallRethrowFnName();
  llvm::Constant *RethrowFn;
  if (!RethrowName.empty())
    RethrowFn = getCatchallRethrowFn(*this, RethrowName);
  else
    RethrowFn = getUnwindResumeOrRethrowFn();

  Builder.CreateCall(RethrowFn, Builder.CreateLoad(getExceptionSlot()))
    ->setDoesNotReturn();
  Builder.CreateUnreachable();

  Builder.restoreIP(SavedIP);

  RethrowBlock = UnwindDest(Unwind, EHStack.stable_end(), 0);
  return RethrowBlock;
}

