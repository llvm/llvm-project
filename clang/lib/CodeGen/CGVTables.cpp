//===--- CGVTables.cpp - Emit LLVM Code for C++ vtables -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ code generation of virtual tables.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "CGCXXABI.h"
#include "CodeGenModule.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/RecordLayout.h"
#include "clang/CodeGen/CGFunctionInfo.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Format.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <algorithm>
#include <cstdio>

using namespace clang;
using namespace CodeGen;

CodeGenVTables::CodeGenVTables(CodeGenModule &CGM)
    : CGM(CGM), VTContext(CGM.getContext().getVTableContext()) {}

llvm::Constant *CodeGenModule::GetAddrOfThunk(GlobalDecl GD, 
                                              const ThunkInfo &Thunk) {
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(GD.getDecl());

  // Compute the mangled name.
  SmallString<256> Name;
  llvm::raw_svector_ostream Out(Name);
  if (const CXXDestructorDecl* DD = dyn_cast<CXXDestructorDecl>(MD))
    getCXXABI().getMangleContext().mangleCXXDtorThunk(DD, GD.getDtorType(),
                                                      Thunk.This, Out);
  else
    getCXXABI().getMangleContext().mangleThunk(MD, Thunk, Out);

  llvm::Type *Ty = getTypes().GetFunctionTypeForVTable(GD);
  return GetOrCreateLLVMFunction(Name, Ty, GD, /*ForVTable=*/true,
                                 /*DontDefer=*/true, /*IsThunk=*/true);
}

static void setThunkVisibility(CodeGenModule &CGM, const CXXMethodDecl *MD,
                               const ThunkInfo &Thunk, llvm::Function *Fn) {
  CGM.setGlobalVisibility(Fn, MD);
}

static void setThunkProperties(CodeGenModule &CGM, const ThunkInfo &Thunk,
                               llvm::Function *ThunkFn, bool ForVTable,
                               GlobalDecl GD) {
  CGM.setFunctionLinkage(GD, ThunkFn);
  CGM.getCXXABI().setThunkLinkage(ThunkFn, ForVTable, GD,
                                  !Thunk.Return.isEmpty());

  // Set the right visibility.
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(GD.getDecl());
  setThunkVisibility(CGM, MD, Thunk, ThunkFn);

  if (CGM.supportsCOMDAT() && ThunkFn->isWeakForLinker())
    ThunkFn->setComdat(CGM.getModule().getOrInsertComdat(ThunkFn->getName()));
}

#ifndef NDEBUG
static bool similar(const ABIArgInfo &infoL, CanQualType typeL,
                    const ABIArgInfo &infoR, CanQualType typeR) {
  return (infoL.getKind() == infoR.getKind() &&
          (typeL == typeR ||
           (isa<PointerType>(typeL) && isa<PointerType>(typeR)) ||
           (isa<ReferenceType>(typeL) && isa<ReferenceType>(typeR))));
}
#endif

static RValue PerformReturnAdjustment(CodeGenFunction &CGF,
                                      QualType ResultType, RValue RV,
                                      const ThunkInfo &Thunk) {
  // Emit the return adjustment.
  bool NullCheckValue = !ResultType->isReferenceType();

  llvm::BasicBlock *AdjustNull = nullptr;
  llvm::BasicBlock *AdjustNotNull = nullptr;
  llvm::BasicBlock *AdjustEnd = nullptr;

  llvm::Value *ReturnValue = RV.getScalarVal();

  if (NullCheckValue) {
    AdjustNull = CGF.createBasicBlock("adjust.null");
    AdjustNotNull = CGF.createBasicBlock("adjust.notnull");
    AdjustEnd = CGF.createBasicBlock("adjust.end");
  
    llvm::Value *IsNull = CGF.Builder.CreateIsNull(ReturnValue);
    CGF.Builder.CreateCondBr(IsNull, AdjustNull, AdjustNotNull);
    CGF.EmitBlock(AdjustNotNull);
  }

  auto ClassDecl = ResultType->getPointeeType()->getAsCXXRecordDecl();
  auto ClassAlign = CGF.CGM.getClassPointerAlignment(ClassDecl);
  ReturnValue = CGF.CGM.getCXXABI().performReturnAdjustment(CGF,
                                            Address(ReturnValue, ClassAlign),
                                            Thunk.Return);

  if (NullCheckValue) {
    CGF.Builder.CreateBr(AdjustEnd);
    CGF.EmitBlock(AdjustNull);
    CGF.Builder.CreateBr(AdjustEnd);
    CGF.EmitBlock(AdjustEnd);
  
    llvm::PHINode *PHI = CGF.Builder.CreatePHI(ReturnValue->getType(), 2);
    PHI->addIncoming(ReturnValue, AdjustNotNull);
    PHI->addIncoming(llvm::Constant::getNullValue(ReturnValue->getType()), 
                     AdjustNull);
    ReturnValue = PHI;
  }
  
  return RValue::get(ReturnValue);
}

// This function does roughly the same thing as GenerateThunk, but in a
// very different way, so that va_start and va_end work correctly.
// FIXME: This function assumes "this" is the first non-sret LLVM argument of
//        a function, and that there is an alloca built in the entry block
//        for all accesses to "this".
// FIXME: This function assumes there is only one "ret" statement per function.
// FIXME: Cloning isn't correct in the presence of indirect goto!
// FIXME: This implementation of thunks bloats codesize by duplicating the
//        function definition.  There are alternatives:
//        1. Add some sort of stub support to LLVM for cases where we can
//           do a this adjustment, then a sibcall.
//        2. We could transform the definition to take a va_list instead of an
//           actual variable argument list, then have the thunks (including a
//           no-op thunk for the regular definition) call va_start/va_end.
//           There's a bit of per-call overhead for this solution, but it's
//           better for codesize if the definition is long.
llvm::Function *
CodeGenFunction::GenerateVarArgsThunk(llvm::Function *Fn,
                                      const CGFunctionInfo &FnInfo,
                                      GlobalDecl GD, const ThunkInfo &Thunk) {
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(GD.getDecl());
  const FunctionProtoType *FPT = MD->getType()->getAs<FunctionProtoType>();
  QualType ResultType = FPT->getReturnType();

  // Get the original function
  assert(FnInfo.isVariadic());
  llvm::Type *Ty = CGM.getTypes().GetFunctionType(FnInfo);
  llvm::Value *Callee = CGM.GetAddrOfFunction(GD, Ty, /*ForVTable=*/true);
  llvm::Function *BaseFn = cast<llvm::Function>(Callee);

  // Clone to thunk.
  llvm::ValueToValueMapTy VMap;
  llvm::Function *NewFn = llvm::CloneFunction(BaseFn, VMap,
                                              /*ModuleLevelChanges=*/false);
  CGM.getModule().getFunctionList().push_back(NewFn);
  Fn->replaceAllUsesWith(NewFn);
  NewFn->takeName(Fn);
  Fn->eraseFromParent();
  Fn = NewFn;

  // "Initialize" CGF (minimally).
  CurFn = Fn;

  // Get the "this" value
  llvm::Function::arg_iterator AI = Fn->arg_begin();
  if (CGM.ReturnTypeUsesSRet(FnInfo))
    ++AI;

  // Find the first store of "this", which will be to the alloca associated
  // with "this".
  Address ThisPtr(&*AI, CGM.getClassPointerAlignment(MD->getParent()));
  llvm::BasicBlock *EntryBB = Fn->begin();
  llvm::Instruction *ThisStore =
      std::find_if(EntryBB->begin(), EntryBB->end(), [&](llvm::Instruction &I) {
    return isa<llvm::StoreInst>(I) && I.getOperand(0) == ThisPtr.getPointer();
  });
  assert(ThisStore && "Store of this should be in entry block?");
  // Adjust "this", if necessary.
  Builder.SetInsertPoint(ThisStore);
  llvm::Value *AdjustedThisPtr =
      CGM.getCXXABI().performThisAdjustment(*this, ThisPtr, Thunk.This);
  ThisStore->setOperand(0, AdjustedThisPtr);

  if (!Thunk.Return.isEmpty()) {
    // Fix up the returned value, if necessary.
    for (llvm::BasicBlock &BB : *Fn) {
      llvm::Instruction *T = BB.getTerminator();
      if (isa<llvm::ReturnInst>(T)) {
        RValue RV = RValue::get(T->getOperand(0));
        T->eraseFromParent();
        Builder.SetInsertPoint(&BB);
        RV = PerformReturnAdjustment(*this, ResultType, RV, Thunk);
        Builder.CreateRet(RV.getScalarVal());
        break;
      }
    }
  }

  return Fn;
}

void CodeGenFunction::StartThunk(llvm::Function *Fn, GlobalDecl GD,
                                 const CGFunctionInfo &FnInfo) {
  assert(!CurGD.getDecl() && "CurGD was already set!");
  CurGD = GD;
  CurFuncIsThunk = true;

  // Build FunctionArgs.
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(GD.getDecl());
  QualType ThisType = MD->getThisType(getContext());
  const FunctionProtoType *FPT = MD->getType()->getAs<FunctionProtoType>();
  QualType ResultType = CGM.getCXXABI().HasThisReturn(GD)
                            ? ThisType
                            : CGM.getCXXABI().hasMostDerivedReturn(GD)
                                  ? CGM.getContext().VoidPtrTy
                                  : FPT->getReturnType();
  FunctionArgList FunctionArgs;

  // Create the implicit 'this' parameter declaration.
  CGM.getCXXABI().buildThisParam(*this, FunctionArgs);

  // Add the rest of the parameters.
  FunctionArgs.append(MD->param_begin(), MD->param_end());

  if (isa<CXXDestructorDecl>(MD))
    CGM.getCXXABI().addImplicitStructorParams(*this, ResultType, FunctionArgs);

  // Start defining the function.
  StartFunction(GlobalDecl(), ResultType, Fn, FnInfo, FunctionArgs,
                MD->getLocation(), MD->getLocation());

  // Since we didn't pass a GlobalDecl to StartFunction, do this ourselves.
  CGM.getCXXABI().EmitInstanceFunctionProlog(*this);
  CXXThisValue = CXXABIThisValue;
  CurCodeDecl = MD;
  CurFuncDecl = MD;
}

void CodeGenFunction::FinishThunk() {
  // Clear these to restore the invariants expected by
  // StartFunction/FinishFunction.
  CurCodeDecl = nullptr;
  CurFuncDecl = nullptr;

  FinishFunction();
}

void CodeGenFunction::EmitCallAndReturnForThunk(llvm::Value *Callee,
                                                const ThunkInfo *Thunk) {
  assert(isa<CXXMethodDecl>(CurGD.getDecl()) &&
         "Please use a new CGF for this thunk");
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(CurGD.getDecl());

  // Adjust the 'this' pointer if necessary
  llvm::Value *AdjustedThisPtr =
    Thunk ? CGM.getCXXABI().performThisAdjustment(
                          *this, LoadCXXThisAddress(), Thunk->This)
          : LoadCXXThis();

  if (CurFnInfo->usesInAlloca()) {
    // We don't handle return adjusting thunks, because they require us to call
    // the copy constructor.  For now, fall through and pretend the return
    // adjustment was empty so we don't crash.
    if (Thunk && !Thunk->Return.isEmpty()) {
      CGM.ErrorUnsupported(
          MD, "non-trivial argument copy for return-adjusting thunk");
    }
    EmitMustTailThunk(MD, AdjustedThisPtr, Callee);
    return;
  }

  // Start building CallArgs.
  CallArgList CallArgs;
  QualType ThisType = MD->getThisType(getContext());
  CallArgs.add(RValue::get(AdjustedThisPtr), ThisType);

  if (isa<CXXDestructorDecl>(MD))
    CGM.getCXXABI().adjustCallArgsForDestructorThunk(*this, CurGD, CallArgs);

  // Add the rest of the arguments.
  for (const ParmVarDecl *PD : MD->params())
    EmitDelegateCallArg(CallArgs, PD, PD->getLocStart());

  const FunctionProtoType *FPT = MD->getType()->getAs<FunctionProtoType>();

#ifndef NDEBUG
  const CGFunctionInfo &CallFnInfo =
    CGM.getTypes().arrangeCXXMethodCall(CallArgs, FPT,
                                       RequiredArgs::forPrototypePlus(FPT, 1));
  assert(CallFnInfo.getRegParm() == CurFnInfo->getRegParm() &&
         CallFnInfo.isNoReturn() == CurFnInfo->isNoReturn() &&
         CallFnInfo.getCallingConvention() == CurFnInfo->getCallingConvention());
  assert(isa<CXXDestructorDecl>(MD) || // ignore dtor return types
         similar(CallFnInfo.getReturnInfo(), CallFnInfo.getReturnType(),
                 CurFnInfo->getReturnInfo(), CurFnInfo->getReturnType()));
  assert(CallFnInfo.arg_size() == CurFnInfo->arg_size());
  for (unsigned i = 0, e = CurFnInfo->arg_size(); i != e; ++i)
    assert(similar(CallFnInfo.arg_begin()[i].info,
                   CallFnInfo.arg_begin()[i].type,
                   CurFnInfo->arg_begin()[i].info,
                   CurFnInfo->arg_begin()[i].type));
#endif

  // Determine whether we have a return value slot to use.
  QualType ResultType = CGM.getCXXABI().HasThisReturn(CurGD)
                            ? ThisType
                            : CGM.getCXXABI().hasMostDerivedReturn(CurGD)
                                  ? CGM.getContext().VoidPtrTy
                                  : FPT->getReturnType();
  ReturnValueSlot Slot;
  if (!ResultType->isVoidType() &&
      CurFnInfo->getReturnInfo().getKind() == ABIArgInfo::Indirect &&
      !hasScalarEvaluationKind(CurFnInfo->getReturnType()))
    Slot = ReturnValueSlot(ReturnValue, ResultType.isVolatileQualified());
  
  // Now emit our call.
  llvm::Instruction *CallOrInvoke;
  RValue RV = EmitCall(*CurFnInfo, Callee, Slot, CallArgs, MD, &CallOrInvoke);

  // Consider return adjustment if we have ThunkInfo.
  if (Thunk && !Thunk->Return.isEmpty())
    RV = PerformReturnAdjustment(*this, ResultType, RV, *Thunk);
  else if (llvm::CallInst* Call = dyn_cast<llvm::CallInst>(CallOrInvoke))
    Call->setTailCallKind(llvm::CallInst::TCK_Tail);

  // Emit return.
  if (!ResultType->isVoidType() && Slot.isNull())
    CGM.getCXXABI().EmitReturnFromThunk(*this, RV, ResultType);

  // Disable the final ARC autorelease.
  AutoreleaseResult = false;

  FinishThunk();
}

void CodeGenFunction::EmitMustTailThunk(const CXXMethodDecl *MD,
                                        llvm::Value *AdjustedThisPtr,
                                        llvm::Value *Callee) {
  // Emitting a musttail call thunk doesn't use any of the CGCall.cpp machinery
  // to translate AST arguments into LLVM IR arguments.  For thunks, we know
  // that the caller prototype more or less matches the callee prototype with
  // the exception of 'this'.
  SmallVector<llvm::Value *, 8> Args;
  for (llvm::Argument &A : CurFn->args())
    Args.push_back(&A);

  // Set the adjusted 'this' pointer.
  const ABIArgInfo &ThisAI = CurFnInfo->arg_begin()->info;
  if (ThisAI.isDirect()) {
    const ABIArgInfo &RetAI = CurFnInfo->getReturnInfo();
    int ThisArgNo = RetAI.isIndirect() && !RetAI.isSRetAfterThis() ? 1 : 0;
    llvm::Type *ThisType = Args[ThisArgNo]->getType();
    if (ThisType != AdjustedThisPtr->getType())
      AdjustedThisPtr = Builder.CreateBitCast(AdjustedThisPtr, ThisType);
    Args[ThisArgNo] = AdjustedThisPtr;
  } else {
    assert(ThisAI.isInAlloca() && "this is passed directly or inalloca");
    Address ThisAddr = GetAddrOfLocalVar(CXXABIThisDecl);
    llvm::Type *ThisType = ThisAddr.getElementType();
    if (ThisType != AdjustedThisPtr->getType())
      AdjustedThisPtr = Builder.CreateBitCast(AdjustedThisPtr, ThisType);
    Builder.CreateStore(AdjustedThisPtr, ThisAddr);
  }

  // Emit the musttail call manually.  Even if the prologue pushed cleanups, we
  // don't actually want to run them.
  llvm::CallInst *Call = Builder.CreateCall(Callee, Args);
  Call->setTailCallKind(llvm::CallInst::TCK_MustTail);

  // Apply the standard set of call attributes.
  unsigned CallingConv;
  CodeGen::AttributeListType AttributeList;
  CGM.ConstructAttributeList(*CurFnInfo, MD, AttributeList, CallingConv,
                             /*AttrOnCallSite=*/true);
  llvm::AttributeSet Attrs =
      llvm::AttributeSet::get(getLLVMContext(), AttributeList);
  Call->setAttributes(Attrs);
  Call->setCallingConv(static_cast<llvm::CallingConv::ID>(CallingConv));

  if (Call->getType()->isVoidTy())
    Builder.CreateRetVoid();
  else
    Builder.CreateRet(Call);

  // Finish the function to maintain CodeGenFunction invariants.
  // FIXME: Don't emit unreachable code.
  EmitBlock(createBasicBlock());
  FinishFunction();
}

void CodeGenFunction::generateThunk(llvm::Function *Fn,
                                    const CGFunctionInfo &FnInfo,
                                    GlobalDecl GD, const ThunkInfo &Thunk) {
  StartThunk(Fn, GD, FnInfo);

  // Get our callee.
  llvm::Type *Ty =
    CGM.getTypes().GetFunctionType(CGM.getTypes().arrangeGlobalDeclaration(GD));
  llvm::Value *Callee = CGM.GetAddrOfFunction(GD, Ty, /*ForVTable=*/true);

  // Make the call and return the result.
  EmitCallAndReturnForThunk(Callee, &Thunk);
}

void CodeGenVTables::emitThunk(GlobalDecl GD, const ThunkInfo &Thunk,
                               bool ForVTable) {
  const CGFunctionInfo &FnInfo = CGM.getTypes().arrangeGlobalDeclaration(GD);

  // FIXME: re-use FnInfo in this computation.
  llvm::Constant *C = CGM.GetAddrOfThunk(GD, Thunk);
  llvm::GlobalValue *Entry;

  // Strip off a bitcast if we got one back.
  if (llvm::ConstantExpr *CE = dyn_cast<llvm::ConstantExpr>(C)) {
    assert(CE->getOpcode() == llvm::Instruction::BitCast);
    Entry = cast<llvm::GlobalValue>(CE->getOperand(0));
  } else {
    Entry = cast<llvm::GlobalValue>(C);
  }

  // There's already a declaration with the same name, check if it has the same
  // type or if we need to replace it.
  if (Entry->getType()->getElementType() !=
      CGM.getTypes().GetFunctionTypeForVTable(GD)) {
    llvm::GlobalValue *OldThunkFn = Entry;

    // If the types mismatch then we have to rewrite the definition.
    assert(OldThunkFn->isDeclaration() &&
           "Shouldn't replace non-declaration");

    // Remove the name from the old thunk function and get a new thunk.
    OldThunkFn->setName(StringRef());
    Entry = cast<llvm::GlobalValue>(CGM.GetAddrOfThunk(GD, Thunk));
    
    // If needed, replace the old thunk with a bitcast.
    if (!OldThunkFn->use_empty()) {
      llvm::Constant *NewPtrForOldDecl =
        llvm::ConstantExpr::getBitCast(Entry, OldThunkFn->getType());
      OldThunkFn->replaceAllUsesWith(NewPtrForOldDecl);
    }
    
    // Remove the old thunk.
    OldThunkFn->eraseFromParent();
  }

  llvm::Function *ThunkFn = cast<llvm::Function>(Entry);
  bool ABIHasKeyFunctions = CGM.getTarget().getCXXABI().hasKeyFunctions();
  bool UseAvailableExternallyLinkage = ForVTable && ABIHasKeyFunctions;

  if (!ThunkFn->isDeclaration()) {
    if (!ABIHasKeyFunctions || UseAvailableExternallyLinkage) {
      // There is already a thunk emitted for this function, do nothing.
      return;
    }

    setThunkProperties(CGM, Thunk, ThunkFn, ForVTable, GD);
    return;
  }

  CGM.SetLLVMFunctionAttributesForDefinition(GD.getDecl(), ThunkFn);

  if (ThunkFn->isVarArg()) {
    // Varargs thunks are special; we can't just generate a call because
    // we can't copy the varargs.  Our implementation is rather
    // expensive/sucky at the moment, so don't generate the thunk unless
    // we have to.
    // FIXME: Do something better here; GenerateVarArgsThunk is extremely ugly.
    if (UseAvailableExternallyLinkage)
      return;
    ThunkFn =
        CodeGenFunction(CGM).GenerateVarArgsThunk(ThunkFn, FnInfo, GD, Thunk);
  } else {
    // Normal thunk body generation.
    CodeGenFunction(CGM).generateThunk(ThunkFn, FnInfo, GD, Thunk);
  }

  setThunkProperties(CGM, Thunk, ThunkFn, ForVTable, GD);
}

void CodeGenVTables::maybeEmitThunkForVTable(GlobalDecl GD,
                                             const ThunkInfo &Thunk) {
  // If the ABI has key functions, only the TU with the key function should emit
  // the thunk. However, we can allow inlining of thunks if we emit them with
  // available_externally linkage together with vtables when optimizations are
  // enabled.
  if (CGM.getTarget().getCXXABI().hasKeyFunctions() &&
      !CGM.getCodeGenOpts().OptimizationLevel)
    return;

  // We can't emit thunks for member functions with incomplete types.
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(GD.getDecl());
  if (!CGM.getTypes().isFuncTypeConvertible(
           MD->getType()->castAs<FunctionType>()))
    return;

  emitThunk(GD, Thunk, /*ForVTable=*/true);
}

void CodeGenVTables::EmitThunks(GlobalDecl GD)
{
  const CXXMethodDecl *MD = 
    cast<CXXMethodDecl>(GD.getDecl())->getCanonicalDecl();

  // We don't need to generate thunks for the base destructor.
  if (isa<CXXDestructorDecl>(MD) && GD.getDtorType() == Dtor_Base)
    return;

  const VTableContextBase::ThunkInfoVectorTy *ThunkInfoVector =
      VTContext->getThunkInfo(GD);

  if (!ThunkInfoVector)
    return;

  for (const ThunkInfo& Thunk : *ThunkInfoVector)
    emitThunk(GD, Thunk, /*ForVTable=*/false);
}

llvm::Constant *CodeGenVTables::CreateVTableInitializer(
    const CXXRecordDecl *RD, const VTableComponent *Components,
    unsigned NumComponents, const VTableLayout::VTableThunkTy *VTableThunks,
    unsigned NumVTableThunks, llvm::Constant *RTTI) {
  SmallVector<llvm::Constant *, 64> Inits;

  llvm::Type *Int8PtrTy = CGM.Int8PtrTy;
  
  llvm::Type *PtrDiffTy = 
    CGM.getTypes().ConvertType(CGM.getContext().getPointerDiffType());

  unsigned NextVTableThunkIndex = 0;

  llvm::Constant *PureVirtualFn = nullptr, *DeletedVirtualFn = nullptr;

  for (unsigned I = 0; I != NumComponents; ++I) {
    VTableComponent Component = Components[I];

    llvm::Constant *Init = nullptr;

    switch (Component.getKind()) {
    case VTableComponent::CK_VCallOffset:
      Init = llvm::ConstantInt::get(PtrDiffTy, 
                                    Component.getVCallOffset().getQuantity());
      Init = llvm::ConstantExpr::getIntToPtr(Init, Int8PtrTy);
      break;
    case VTableComponent::CK_VBaseOffset:
      Init = llvm::ConstantInt::get(PtrDiffTy, 
                                    Component.getVBaseOffset().getQuantity());
      Init = llvm::ConstantExpr::getIntToPtr(Init, Int8PtrTy);
      break;
    case VTableComponent::CK_OffsetToTop:
      Init = llvm::ConstantInt::get(PtrDiffTy, 
                                    Component.getOffsetToTop().getQuantity());
      Init = llvm::ConstantExpr::getIntToPtr(Init, Int8PtrTy);
      break;
    case VTableComponent::CK_RTTI:
      Init = llvm::ConstantExpr::getBitCast(RTTI, Int8PtrTy);
      break;
    case VTableComponent::CK_FunctionPointer:
    case VTableComponent::CK_CompleteDtorPointer:
    case VTableComponent::CK_DeletingDtorPointer: {
      GlobalDecl GD;
      
      // Get the right global decl.
      switch (Component.getKind()) {
      default:
        llvm_unreachable("Unexpected vtable component kind");
      case VTableComponent::CK_FunctionPointer:
        GD = Component.getFunctionDecl();
        break;
      case VTableComponent::CK_CompleteDtorPointer:
        GD = GlobalDecl(Component.getDestructorDecl(), Dtor_Complete);
        break;
      case VTableComponent::CK_DeletingDtorPointer:
        GD = GlobalDecl(Component.getDestructorDecl(), Dtor_Deleting);
        break;
      }

      if (cast<CXXMethodDecl>(GD.getDecl())->isPure()) {
        // We have a pure virtual member function.
        if (!PureVirtualFn) {
          llvm::FunctionType *Ty = 
            llvm::FunctionType::get(CGM.VoidTy, /*isVarArg=*/false);
          StringRef PureCallName = CGM.getCXXABI().GetPureVirtualCallName();
          PureVirtualFn = CGM.CreateRuntimeFunction(Ty, PureCallName);
          PureVirtualFn = llvm::ConstantExpr::getBitCast(PureVirtualFn,
                                                         CGM.Int8PtrTy);
        }
        Init = PureVirtualFn;
      } else if (cast<CXXMethodDecl>(GD.getDecl())->isDeleted()) {
        if (!DeletedVirtualFn) {
          llvm::FunctionType *Ty =
            llvm::FunctionType::get(CGM.VoidTy, /*isVarArg=*/false);
          StringRef DeletedCallName =
            CGM.getCXXABI().GetDeletedVirtualCallName();
          DeletedVirtualFn = CGM.CreateRuntimeFunction(Ty, DeletedCallName);
          DeletedVirtualFn = llvm::ConstantExpr::getBitCast(DeletedVirtualFn,
                                                         CGM.Int8PtrTy);
        }
        Init = DeletedVirtualFn;
      } else {
        // Check if we should use a thunk.
        if (NextVTableThunkIndex < NumVTableThunks &&
            VTableThunks[NextVTableThunkIndex].first == I) {
          const ThunkInfo &Thunk = VTableThunks[NextVTableThunkIndex].second;
        
          maybeEmitThunkForVTable(GD, Thunk);
          Init = CGM.GetAddrOfThunk(GD, Thunk);

          NextVTableThunkIndex++;
        } else {
          llvm::Type *Ty = CGM.getTypes().GetFunctionTypeForVTable(GD);
        
          Init = CGM.GetAddrOfFunction(GD, Ty, /*ForVTable=*/true);
        }

        Init = llvm::ConstantExpr::getBitCast(Init, Int8PtrTy);
      }
      break;
    }

    case VTableComponent::CK_UnusedFunctionPointer:
      Init = llvm::ConstantExpr::getNullValue(Int8PtrTy);
      break;
    };
    
    Inits.push_back(Init);
  }
  
  llvm::ArrayType *ArrayType = llvm::ArrayType::get(Int8PtrTy, NumComponents);
  return llvm::ConstantArray::get(ArrayType, Inits);
}

llvm::GlobalVariable *
CodeGenVTables::GenerateConstructionVTable(const CXXRecordDecl *RD, 
                                      const BaseSubobject &Base, 
                                      bool BaseIsVirtual, 
                                   llvm::GlobalVariable::LinkageTypes Linkage,
                                      VTableAddressPointsMapTy& AddressPoints) {
  if (CGDebugInfo *DI = CGM.getModuleDebugInfo())
    DI->completeClassData(Base.getBase());

  std::unique_ptr<VTableLayout> VTLayout(
      getItaniumVTableContext().createConstructionVTableLayout(
          Base.getBase(), Base.getBaseOffset(), BaseIsVirtual, RD));

  // Add the address points.
  AddressPoints = VTLayout->getAddressPoints();

  // Get the mangled construction vtable name.
  SmallString<256> OutName;
  llvm::raw_svector_ostream Out(OutName);
  cast<ItaniumMangleContext>(CGM.getCXXABI().getMangleContext())
      .mangleCXXCtorVTable(RD, Base.getBaseOffset().getQuantity(),
                           Base.getBase(), Out);
  StringRef Name = OutName.str();

  llvm::ArrayType *ArrayType = 
    llvm::ArrayType::get(CGM.Int8PtrTy, VTLayout->getNumVTableComponents());

  // Construction vtable symbols are not part of the Itanium ABI, so we cannot
  // guarantee that they actually will be available externally. Instead, when
  // emitting an available_externally VTT, we provide references to an internal
  // linkage construction vtable. The ABI only requires complete-object vtables
  // to be the same for all instances of a type, not construction vtables.
  if (Linkage == llvm::GlobalVariable::AvailableExternallyLinkage)
    Linkage = llvm::GlobalVariable::InternalLinkage;

  // Create the variable that will hold the construction vtable.
  llvm::GlobalVariable *VTable = 
    CGM.CreateOrReplaceCXXRuntimeVariable(Name, ArrayType, Linkage);
  CGM.setGlobalVisibility(VTable, RD);

  // V-tables are always unnamed_addr.
  VTable->setUnnamedAddr(true);

  llvm::Constant *RTTI = CGM.GetAddrOfRTTIDescriptor(
      CGM.getContext().getTagDeclType(Base.getBase()));

  // Create and set the initializer.
  llvm::Constant *Init = CreateVTableInitializer(
      Base.getBase(), VTLayout->vtable_component_begin(),
      VTLayout->getNumVTableComponents(), VTLayout->vtable_thunk_begin(),
      VTLayout->getNumVTableThunks(), RTTI);
  VTable->setInitializer(Init);
  
  CGM.EmitVTableBitSetEntries(VTable, *VTLayout.get());

  return VTable;
}

static bool shouldEmitAvailableExternallyVTable(const CodeGenModule &CGM,
                                                const CXXRecordDecl *RD) {
  return CGM.getCodeGenOpts().OptimizationLevel > 0 &&
         CGM.getCXXABI().canSpeculativelyEmitVTable(RD);
}

/// Compute the required linkage of the v-table for the given class.
///
/// Note that we only call this at the end of the translation unit.
llvm::GlobalVariable::LinkageTypes 
CodeGenModule::getVTableLinkage(const CXXRecordDecl *RD) {
  if (!RD->isExternallyVisible())
    return llvm::GlobalVariable::InternalLinkage;

  // We're at the end of the translation unit, so the current key
  // function is fully correct.
  const CXXMethodDecl *keyFunction = Context.getCurrentKeyFunction(RD);
  if (keyFunction && !RD->hasAttr<DLLImportAttr>()) {
    // If this class has a key function, use that to determine the
    // linkage of the vtable.
    const FunctionDecl *def = nullptr;
    if (keyFunction->hasBody(def))
      keyFunction = cast<CXXMethodDecl>(def);
    
    switch (keyFunction->getTemplateSpecializationKind()) {
      case TSK_Undeclared:
      case TSK_ExplicitSpecialization:
        assert((def || CodeGenOpts.OptimizationLevel > 0) &&
               "Shouldn't query vtable linkage without key function or "
               "optimizations");
        if (!def && CodeGenOpts.OptimizationLevel > 0)
          return llvm::GlobalVariable::AvailableExternallyLinkage;

        if (keyFunction->isInlined())
          return !Context.getLangOpts().AppleKext ?
                   llvm::GlobalVariable::LinkOnceODRLinkage :
                   llvm::Function::InternalLinkage;
        
        return llvm::GlobalVariable::ExternalLinkage;

      case TSK_ImplicitInstantiation:
        return !Context.getLangOpts().AppleKext ?
                 llvm::GlobalVariable::LinkOnceODRLinkage :
                 llvm::Function::InternalLinkage;

      case TSK_ExplicitInstantiationDefinition:
        return !Context.getLangOpts().AppleKext ?
                 llvm::GlobalVariable::WeakODRLinkage :
                 llvm::Function::InternalLinkage;
  
      case TSK_ExplicitInstantiationDeclaration:
        llvm_unreachable("Should not have been asked to emit this");
    }
  }

  // -fapple-kext mode does not support weak linkage, so we must use
  // internal linkage.
  if (Context.getLangOpts().AppleKext)
    return llvm::Function::InternalLinkage;

  llvm::GlobalVariable::LinkageTypes DiscardableODRLinkage =
      llvm::GlobalValue::LinkOnceODRLinkage;
  llvm::GlobalVariable::LinkageTypes NonDiscardableODRLinkage =
      llvm::GlobalValue::WeakODRLinkage;
  if (RD->hasAttr<DLLExportAttr>()) {
    // Cannot discard exported vtables.
    DiscardableODRLinkage = NonDiscardableODRLinkage;
  } else if (RD->hasAttr<DLLImportAttr>()) {
    // Imported vtables are available externally.
    DiscardableODRLinkage = llvm::GlobalVariable::AvailableExternallyLinkage;
    NonDiscardableODRLinkage = llvm::GlobalVariable::AvailableExternallyLinkage;
  }

  switch (RD->getTemplateSpecializationKind()) {
    case TSK_Undeclared:
    case TSK_ExplicitSpecialization:
    case TSK_ImplicitInstantiation:
      return DiscardableODRLinkage;

    case TSK_ExplicitInstantiationDeclaration:
      return shouldEmitAvailableExternallyVTable(*this, RD)
                 ? llvm::GlobalVariable::AvailableExternallyLinkage
                 : llvm::GlobalVariable::ExternalLinkage;

    case TSK_ExplicitInstantiationDefinition:
      return NonDiscardableODRLinkage;
  }

  llvm_unreachable("Invalid TemplateSpecializationKind!");
}

/// This is a callback from Sema to tell us that that a particular v-table is
/// required to be emitted in this translation unit.
///
/// This is only called for vtables that _must_ be emitted (mainly due to key
/// functions).  For weak vtables, CodeGen tracks when they are needed and
/// emits them as-needed.
void CodeGenModule::EmitVTable(CXXRecordDecl *theClass) {
  VTables.GenerateClassData(theClass);
}

void 
CodeGenVTables::GenerateClassData(const CXXRecordDecl *RD) {
  if (CGDebugInfo *DI = CGM.getModuleDebugInfo())
    DI->completeClassData(RD);

  if (RD->getNumVBases())
    CGM.getCXXABI().emitVirtualInheritanceTables(RD);

  CGM.getCXXABI().emitVTableDefinitions(*this, RD);
}

/// At this point in the translation unit, does it appear that can we
/// rely on the vtable being defined elsewhere in the program?
///
/// The response is really only definitive when called at the end of
/// the translation unit.
///
/// The only semantic restriction here is that the object file should
/// not contain a v-table definition when that v-table is defined
/// strongly elsewhere.  Otherwise, we'd just like to avoid emitting
/// v-tables when unnecessary.
bool CodeGenVTables::isVTableExternal(const CXXRecordDecl *RD) {
  assert(RD->isDynamicClass() && "Non-dynamic classes have no VTable.");

  // If we have an explicit instantiation declaration (and not a
  // definition), the v-table is defined elsewhere.
  TemplateSpecializationKind TSK = RD->getTemplateSpecializationKind();
  if (TSK == TSK_ExplicitInstantiationDeclaration)
    return true;

  // Otherwise, if the class is an instantiated template, the
  // v-table must be defined here.
  if (TSK == TSK_ImplicitInstantiation ||
      TSK == TSK_ExplicitInstantiationDefinition)
    return false;

  // Otherwise, if the class doesn't have a key function (possibly
  // anymore), the v-table must be defined here.
  const CXXMethodDecl *keyFunction = CGM.getContext().getCurrentKeyFunction(RD);
  if (!keyFunction)
    return false;

  // Otherwise, if we don't have a definition of the key function, the
  // v-table must be defined somewhere else.
  return !keyFunction->hasBody();
}

/// Given that we're currently at the end of the translation unit, and
/// we've emitted a reference to the v-table for this class, should
/// we define that v-table?
static bool shouldEmitVTableAtEndOfTranslationUnit(CodeGenModule &CGM,
                                                   const CXXRecordDecl *RD) {
  // If vtable is internal then it has to be done.
  if (!CGM.getVTables().isVTableExternal(RD))
    return true;

  // If it's external then maybe we will need it as available_externally.
  return shouldEmitAvailableExternallyVTable(CGM, RD);
}

/// Given that at some point we emitted a reference to one or more
/// v-tables, and that we are now at the end of the translation unit,
/// decide whether we should emit them.
void CodeGenModule::EmitDeferredVTables() {
#ifndef NDEBUG
  // Remember the size of DeferredVTables, because we're going to assume
  // that this entire operation doesn't modify it.
  size_t savedSize = DeferredVTables.size();
#endif

  for (const CXXRecordDecl *RD : DeferredVTables)
    if (shouldEmitVTableAtEndOfTranslationUnit(*this, RD))
      VTables.GenerateClassData(RD);

  assert(savedSize == DeferredVTables.size() &&
         "deferred extra v-tables during v-table emission?");
  DeferredVTables.clear();
}

bool CodeGenModule::IsCFIBlacklistedRecord(const CXXRecordDecl *RD) {
  if (RD->hasAttr<UuidAttr>() &&
      getContext().getSanitizerBlacklist().isBlacklistedType("attr:uuid"))
    return true;

  return getContext().getSanitizerBlacklist().isBlacklistedType(
      RD->getQualifiedNameAsString());
}

void CodeGenModule::EmitVTableBitSetEntries(llvm::GlobalVariable *VTable,
                                            const VTableLayout &VTLayout) {
  if (!LangOpts.Sanitize.has(SanitizerKind::CFIVCall) &&
      !LangOpts.Sanitize.has(SanitizerKind::CFINVCall) &&
      !LangOpts.Sanitize.has(SanitizerKind::CFIDerivedCast) &&
      !LangOpts.Sanitize.has(SanitizerKind::CFIUnrelatedCast))
    return;

  CharUnits PointerWidth =
      Context.toCharUnitsFromBits(Context.getTargetInfo().getPointerWidth(0));

  typedef std::pair<const CXXRecordDecl *, unsigned> BSEntry;
  std::vector<BSEntry> BitsetEntries;
  // Create a bit set entry for each address point.
  for (auto &&AP : VTLayout.getAddressPoints()) {
    if (IsCFIBlacklistedRecord(AP.first.getBase()))
      continue;

    BitsetEntries.push_back(std::make_pair(AP.first.getBase(), AP.second));
  }

  // Sort the bit set entries for determinism.
  std::sort(BitsetEntries.begin(), BitsetEntries.end(),
            [this](const BSEntry &E1, const BSEntry &E2) {
    if (&E1 == &E2)
      return false;

    std::string S1;
    llvm::raw_string_ostream O1(S1);
    getCXXABI().getMangleContext().mangleTypeName(
        QualType(E1.first->getTypeForDecl(), 0), O1);
    O1.flush();

    std::string S2;
    llvm::raw_string_ostream O2(S2);
    getCXXABI().getMangleContext().mangleTypeName(
        QualType(E2.first->getTypeForDecl(), 0), O2);
    O2.flush();

    if (S1 < S2)
      return true;
    if (S1 != S2)
      return false;

    return E1.second < E2.second;
  });

  llvm::NamedMDNode *BitsetsMD =
      getModule().getOrInsertNamedMetadata("llvm.bitsets");
  for (auto BitsetEntry : BitsetEntries)
    BitsetsMD->addOperand(CreateVTableBitSetEntry(
        VTable, PointerWidth * BitsetEntry.second, BitsetEntry.first));
}
