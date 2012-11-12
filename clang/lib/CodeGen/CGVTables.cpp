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

#include "CodeGenModule.h"
#include "CodeGenFunction.h"
#include "CGCXXABI.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/RecordLayout.h"
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
  : CGM(CGM), VTContext(CGM.getContext()) { }

bool CodeGenVTables::ShouldEmitVTableInThisTU(const CXXRecordDecl *RD) {
  assert(RD->isDynamicClass() && "Non dynamic classes have no VTable.");

  TemplateSpecializationKind TSK = RD->getTemplateSpecializationKind();
  if (TSK == TSK_ExplicitInstantiationDeclaration)
    return false;

  const CXXMethodDecl *KeyFunction = CGM.getContext().getKeyFunction(RD);
  if (!KeyFunction)
    return true;

  // Itanium C++ ABI, 5.2.6 Instantiated Templates:
  //    An instantiation of a class template requires:
  //        - In the object where instantiated, the virtual table...
  if (TSK == TSK_ImplicitInstantiation ||
      TSK == TSK_ExplicitInstantiationDefinition)
    return true;

  // If we're building with optimization, we always emit VTables since that
  // allows for virtual function calls to be devirtualized.
  // (We don't want to do this in -fapple-kext mode however).
  if (CGM.getCodeGenOpts().OptimizationLevel && !CGM.getLangOpts().AppleKext)
    return true;

  return KeyFunction->hasBody();
}

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
  Out.flush();

  llvm::Type *Ty = getTypes().GetFunctionTypeForVTable(GD);
  return GetOrCreateLLVMFunction(Name, Ty, GD, /*ForVTable=*/true);
}

static llvm::Value *PerformTypeAdjustment(CodeGenFunction &CGF,
                                          llvm::Value *Ptr,
                                          int64_t NonVirtualAdjustment,
                                          int64_t VirtualAdjustment,
                                          bool IsReturnAdjustment) {
  if (!NonVirtualAdjustment && !VirtualAdjustment)
    return Ptr;

  llvm::Type *Int8PtrTy = CGF.Int8PtrTy;
  llvm::Value *V = CGF.Builder.CreateBitCast(Ptr, Int8PtrTy);

  if (NonVirtualAdjustment && !IsReturnAdjustment) {
    // Perform the non-virtual adjustment for a base-to-derived cast.
    V = CGF.Builder.CreateConstInBoundsGEP1_64(V, NonVirtualAdjustment);
  }

  if (VirtualAdjustment) {
    llvm::Type *PtrDiffTy = 
      CGF.ConvertType(CGF.getContext().getPointerDiffType());

    // Perform the virtual adjustment.
    llvm::Value *VTablePtrPtr = 
      CGF.Builder.CreateBitCast(V, Int8PtrTy->getPointerTo());
    
    llvm::Value *VTablePtr = CGF.Builder.CreateLoad(VTablePtrPtr);
  
    llvm::Value *OffsetPtr =
      CGF.Builder.CreateConstInBoundsGEP1_64(VTablePtr, VirtualAdjustment);
    
    OffsetPtr = CGF.Builder.CreateBitCast(OffsetPtr, PtrDiffTy->getPointerTo());
    
    // Load the adjustment offset from the vtable.
    llvm::Value *Offset = CGF.Builder.CreateLoad(OffsetPtr);
    
    // Adjust our pointer.
    V = CGF.Builder.CreateInBoundsGEP(V, Offset);
  }

  if (NonVirtualAdjustment && IsReturnAdjustment) {
    // Perform the non-virtual adjustment for a derived-to-base cast.
    V = CGF.Builder.CreateConstInBoundsGEP1_64(V, NonVirtualAdjustment);
  }

  // Cast back to the original type.
  return CGF.Builder.CreateBitCast(V, Ptr->getType());
}

static void setThunkVisibility(CodeGenModule &CGM, const CXXMethodDecl *MD,
                               const ThunkInfo &Thunk, llvm::Function *Fn) {
  CGM.setGlobalVisibility(Fn, MD);

  if (!CGM.getCodeGenOpts().HiddenWeakVTables)
    return;

  // If the thunk has weak/linkonce linkage, but the function must be
  // emitted in every translation unit that references it, then we can
  // emit its thunks with hidden visibility, since its thunks must be
  // emitted when the function is.

  // This follows CodeGenModule::setTypeVisibility; see the comments
  // there for explanation.

  if ((Fn->getLinkage() != llvm::GlobalVariable::LinkOnceODRLinkage &&
       Fn->getLinkage() != llvm::GlobalVariable::WeakODRLinkage) ||
      Fn->getVisibility() != llvm::GlobalVariable::DefaultVisibility)
    return;

  if (MD->getExplicitVisibility())
    return;

  switch (MD->getTemplateSpecializationKind()) {
  case TSK_ExplicitInstantiationDefinition:
  case TSK_ExplicitInstantiationDeclaration:
    return;

  case TSK_Undeclared:
    break;

  case TSK_ExplicitSpecialization:
  case TSK_ImplicitInstantiation:
    return;
    break;
  }

  // If there's an explicit definition, and that definition is
  // out-of-line, then we can't assume that all users will have a
  // definition to emit.
  const FunctionDecl *Def = 0;
  if (MD->hasBody(Def) && Def->isOutOfLine())
    return;

  Fn->setVisibility(llvm::GlobalValue::HiddenVisibility);
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
  
  llvm::BasicBlock *AdjustNull = 0;
  llvm::BasicBlock *AdjustNotNull = 0;
  llvm::BasicBlock *AdjustEnd = 0;
  
  llvm::Value *ReturnValue = RV.getScalarVal();

  if (NullCheckValue) {
    AdjustNull = CGF.createBasicBlock("adjust.null");
    AdjustNotNull = CGF.createBasicBlock("adjust.notnull");
    AdjustEnd = CGF.createBasicBlock("adjust.end");
  
    llvm::Value *IsNull = CGF.Builder.CreateIsNull(ReturnValue);
    CGF.Builder.CreateCondBr(IsNull, AdjustNull, AdjustNotNull);
    CGF.EmitBlock(AdjustNotNull);
  }
  
  ReturnValue = PerformTypeAdjustment(CGF, ReturnValue, 
                                      Thunk.Return.NonVirtual, 
                                      Thunk.Return.VBaseOffsetOffset,
                                      /*IsReturnAdjustment*/true);
  
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
void CodeGenFunction::GenerateVarArgsThunk(
                                      llvm::Function *Fn,
                                      const CGFunctionInfo &FnInfo,
                                      GlobalDecl GD, const ThunkInfo &Thunk) {
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(GD.getDecl());
  const FunctionProtoType *FPT = MD->getType()->getAs<FunctionProtoType>();
  QualType ResultType = FPT->getResultType();

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
  llvm::Value *ThisPtr = &*AI;
  llvm::BasicBlock *EntryBB = Fn->begin();
  llvm::Instruction *ThisStore = 0;
  for (llvm::BasicBlock::iterator I = EntryBB->begin(), E = EntryBB->end();
       I != E; I++) {
    if (isa<llvm::StoreInst>(I) && I->getOperand(0) == ThisPtr) {
      ThisStore = cast<llvm::StoreInst>(I);
      break;
    }
  }
  assert(ThisStore && "Store of this should be in entry block?");
  // Adjust "this", if necessary.
  Builder.SetInsertPoint(ThisStore);
  llvm::Value *AdjustedThisPtr = 
    PerformTypeAdjustment(*this, ThisPtr, 
                          Thunk.This.NonVirtual, 
                          Thunk.This.VCallOffsetOffset,
                          /*IsReturnAdjustment*/false);
  ThisStore->setOperand(0, AdjustedThisPtr);

  if (!Thunk.Return.isEmpty()) {
    // Fix up the returned value, if necessary.
    for (llvm::Function::iterator I = Fn->begin(), E = Fn->end(); I != E; I++) {
      llvm::Instruction *T = I->getTerminator();
      if (isa<llvm::ReturnInst>(T)) {
        RValue RV = RValue::get(T->getOperand(0));
        T->eraseFromParent();
        Builder.SetInsertPoint(&*I);
        RV = PerformReturnAdjustment(*this, ResultType, RV, Thunk);
        Builder.CreateRet(RV.getScalarVal());
        break;
      }
    }
  }
}

void CodeGenFunction::GenerateThunk(llvm::Function *Fn,
                                    const CGFunctionInfo &FnInfo,
                                    GlobalDecl GD, const ThunkInfo &Thunk) {
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(GD.getDecl());
  const FunctionProtoType *FPT = MD->getType()->getAs<FunctionProtoType>();
  QualType ResultType = FPT->getResultType();
  QualType ThisType = MD->getThisType(getContext());

  FunctionArgList FunctionArgs;

  // FIXME: It would be nice if more of this code could be shared with 
  // CodeGenFunction::GenerateCode.

  // Create the implicit 'this' parameter declaration.
  CurGD = GD;
  CGM.getCXXABI().BuildInstanceFunctionParams(*this, ResultType, FunctionArgs);

  // Add the rest of the parameters.
  for (FunctionDecl::param_const_iterator I = MD->param_begin(),
       E = MD->param_end(); I != E; ++I) {
    ParmVarDecl *Param = *I;
    
    FunctionArgs.push_back(Param);
  }

  // Initialize debug info if needed.
  maybeInitializeDebugInfo();

  StartFunction(GlobalDecl(), ResultType, Fn, FnInfo, FunctionArgs,
                SourceLocation());

  CGM.getCXXABI().EmitInstanceFunctionProlog(*this);
  CXXThisValue = CXXABIThisValue;

  // Adjust the 'this' pointer if necessary.
  llvm::Value *AdjustedThisPtr = 
    PerformTypeAdjustment(*this, LoadCXXThis(), 
                          Thunk.This.NonVirtual, 
                          Thunk.This.VCallOffsetOffset,
                          /*IsReturnAdjustment*/false);
  
  CallArgList CallArgs;
  
  // Add our adjusted 'this' pointer.
  CallArgs.add(RValue::get(AdjustedThisPtr), ThisType);

  // Add the rest of the parameters.
  for (FunctionDecl::param_const_iterator I = MD->param_begin(),
       E = MD->param_end(); I != E; ++I) {
    ParmVarDecl *param = *I;
    EmitDelegateCallArg(CallArgs, param);
  }

  // Get our callee.
  llvm::Type *Ty =
    CGM.getTypes().GetFunctionType(CGM.getTypes().arrangeGlobalDeclaration(GD));
  llvm::Value *Callee = CGM.GetAddrOfFunction(GD, Ty, /*ForVTable=*/true);

#ifndef NDEBUG
  const CGFunctionInfo &CallFnInfo =
    CGM.getTypes().arrangeCXXMethodCall(CallArgs, FPT,
                                       RequiredArgs::forPrototypePlus(FPT, 1));
  assert(CallFnInfo.getRegParm() == FnInfo.getRegParm() &&
         CallFnInfo.isNoReturn() == FnInfo.isNoReturn() &&
         CallFnInfo.getCallingConvention() == FnInfo.getCallingConvention());
  assert(isa<CXXDestructorDecl>(MD) || // ignore dtor return types
         similar(CallFnInfo.getReturnInfo(), CallFnInfo.getReturnType(),
                 FnInfo.getReturnInfo(), FnInfo.getReturnType()));
  assert(CallFnInfo.arg_size() == FnInfo.arg_size());
  for (unsigned i = 0, e = FnInfo.arg_size(); i != e; ++i)
    assert(similar(CallFnInfo.arg_begin()[i].info,
                   CallFnInfo.arg_begin()[i].type,
                   FnInfo.arg_begin()[i].info, FnInfo.arg_begin()[i].type));
#endif
  
  // Determine whether we have a return value slot to use.
  ReturnValueSlot Slot;
  if (!ResultType->isVoidType() &&
      FnInfo.getReturnInfo().getKind() == ABIArgInfo::Indirect &&
      hasAggregateLLVMType(CurFnInfo->getReturnType()))
    Slot = ReturnValueSlot(ReturnValue, ResultType.isVolatileQualified());
  
  // Now emit our call.
  RValue RV = EmitCall(FnInfo, Callee, Slot, CallArgs, MD);
  
  if (!Thunk.Return.isEmpty())
    RV = PerformReturnAdjustment(*this, ResultType, RV, Thunk);

  if (!ResultType->isVoidType() && Slot.isNull())
    CGM.getCXXABI().EmitReturnFromThunk(*this, RV, ResultType);

  // Disable the final ARC autorelease.
  AutoreleaseResult = false;

  FinishFunction();

  // Set the right linkage.
  CGM.setFunctionLinkage(MD, Fn);
  
  // Set the right visibility.
  setThunkVisibility(CGM, MD, Thunk, Fn);
}

void CodeGenVTables::EmitThunk(GlobalDecl GD, const ThunkInfo &Thunk, 
                               bool UseAvailableExternallyLinkage)
{
  const CGFunctionInfo &FnInfo = CGM.getTypes().arrangeGlobalDeclaration(GD);

  // FIXME: re-use FnInfo in this computation.
  llvm::Constant *Entry = CGM.GetAddrOfThunk(GD, Thunk);
  
  // Strip off a bitcast if we got one back.
  if (llvm::ConstantExpr *CE = dyn_cast<llvm::ConstantExpr>(Entry)) {
    assert(CE->getOpcode() == llvm::Instruction::BitCast);
    Entry = CE->getOperand(0);
  }
  
  // There's already a declaration with the same name, check if it has the same
  // type or if we need to replace it.
  if (cast<llvm::GlobalValue>(Entry)->getType()->getElementType() != 
      CGM.getTypes().GetFunctionTypeForVTable(GD)) {
    llvm::GlobalValue *OldThunkFn = cast<llvm::GlobalValue>(Entry);
    
    // If the types mismatch then we have to rewrite the definition.
    assert(OldThunkFn->isDeclaration() &&
           "Shouldn't replace non-declaration");

    // Remove the name from the old thunk function and get a new thunk.
    OldThunkFn->setName(StringRef());
    Entry = CGM.GetAddrOfThunk(GD, Thunk);
    
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

  if (!ThunkFn->isDeclaration()) {
    if (UseAvailableExternallyLinkage) {
      // There is already a thunk emitted for this function, do nothing.
      return;
    }

    // If a function has a body, it should have available_externally linkage.
    assert(ThunkFn->hasAvailableExternallyLinkage() &&
           "Function should have available_externally linkage!");

    // Change the linkage.
    CGM.setFunctionLinkage(cast<CXXMethodDecl>(GD.getDecl()), ThunkFn);
    return;
  }

  CGM.SetLLVMFunctionAttributesForDefinition(GD.getDecl(), ThunkFn);

  if (ThunkFn->isVarArg()) {
    // Varargs thunks are special; we can't just generate a call because
    // we can't copy the varargs.  Our implementation is rather
    // expensive/sucky at the moment, so don't generate the thunk unless
    // we have to.
    // FIXME: Do something better here; GenerateVarArgsThunk is extremely ugly.
    if (!UseAvailableExternallyLinkage)
      CodeGenFunction(CGM).GenerateVarArgsThunk(ThunkFn, FnInfo, GD, Thunk);
  } else {
    // Normal thunk body generation.
    CodeGenFunction(CGM).GenerateThunk(ThunkFn, FnInfo, GD, Thunk);
  }

  if (UseAvailableExternallyLinkage)
    ThunkFn->setLinkage(llvm::GlobalValue::AvailableExternallyLinkage);
}

void CodeGenVTables::MaybeEmitThunkAvailableExternally(GlobalDecl GD,
                                                       const ThunkInfo &Thunk) {
  // We only want to do this when building with optimizations.
  if (!CGM.getCodeGenOpts().OptimizationLevel)
    return;

  // We can't emit thunks for member functions with incomplete types.
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(GD.getDecl());
  if (!CGM.getTypes().isFuncTypeConvertible(
                                cast<FunctionType>(MD->getType().getTypePtr())))
    return;

  EmitThunk(GD, Thunk, /*UseAvailableExternallyLinkage=*/true);
}

void CodeGenVTables::EmitThunks(GlobalDecl GD)
{
  const CXXMethodDecl *MD = 
    cast<CXXMethodDecl>(GD.getDecl())->getCanonicalDecl();

  // We don't need to generate thunks for the base destructor.
  if (isa<CXXDestructorDecl>(MD) && GD.getDtorType() == Dtor_Base)
    return;

  const VTableContext::ThunkInfoVectorTy *ThunkInfoVector =
    VTContext.getThunkInfo(MD);
  if (!ThunkInfoVector)
    return;

  for (unsigned I = 0, E = ThunkInfoVector->size(); I != E; ++I)
    EmitThunk(GD, (*ThunkInfoVector)[I],
              /*UseAvailableExternallyLinkage=*/false);
}

llvm::Constant *
CodeGenVTables::CreateVTableInitializer(const CXXRecordDecl *RD,
                                        const VTableComponent *Components, 
                                        unsigned NumComponents,
                                const VTableLayout::VTableThunkTy *VTableThunks,
                                        unsigned NumVTableThunks) {
  SmallVector<llvm::Constant *, 64> Inits;

  llvm::Type *Int8PtrTy = CGM.Int8PtrTy;
  
  llvm::Type *PtrDiffTy = 
    CGM.getTypes().ConvertType(CGM.getContext().getPointerDiffType());

  QualType ClassType = CGM.getContext().getTagDeclType(RD);
  llvm::Constant *RTTI = CGM.GetAddrOfRTTIDescriptor(ClassType);
  
  unsigned NextVTableThunkIndex = 0;
  
  llvm::Constant *PureVirtualFn = 0, *DeletedVirtualFn = 0;

  for (unsigned I = 0; I != NumComponents; ++I) {
    VTableComponent Component = Components[I];

    llvm::Constant *Init = 0;

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
        
          MaybeEmitThunkAvailableExternally(GD, Thunk);
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

llvm::GlobalVariable *CodeGenVTables::GetAddrOfVTable(const CXXRecordDecl *RD) {
  llvm::GlobalVariable *&VTable = VTables[RD];
  if (VTable)
    return VTable;

  // We may need to generate a definition for this vtable.
  if (ShouldEmitVTableInThisTU(RD))
    CGM.DeferredVTables.push_back(RD);

  SmallString<256> OutName;
  llvm::raw_svector_ostream Out(OutName);
  CGM.getCXXABI().getMangleContext().mangleCXXVTable(RD, Out);
  Out.flush();
  StringRef Name = OutName.str();

  llvm::ArrayType *ArrayType = 
    llvm::ArrayType::get(CGM.Int8PtrTy,
                        VTContext.getVTableLayout(RD).getNumVTableComponents());

  VTable =
    CGM.CreateOrReplaceCXXRuntimeVariable(Name, ArrayType, 
                                          llvm::GlobalValue::ExternalLinkage);
  VTable->setUnnamedAddr(true);
  return VTable;
}

void
CodeGenVTables::EmitVTableDefinition(llvm::GlobalVariable *VTable,
                                     llvm::GlobalVariable::LinkageTypes Linkage,
                                     const CXXRecordDecl *RD) {
  const VTableLayout &VTLayout = VTContext.getVTableLayout(RD);

  // Create and set the initializer.
  llvm::Constant *Init = 
    CreateVTableInitializer(RD,
                            VTLayout.vtable_component_begin(),
                            VTLayout.getNumVTableComponents(),
                            VTLayout.vtable_thunk_begin(),
                            VTLayout.getNumVTableThunks());
  VTable->setInitializer(Init);
  
  // Set the correct linkage.
  VTable->setLinkage(Linkage);
  
  // Set the right visibility.
  CGM.setTypeVisibility(VTable, RD, CodeGenModule::TVK_ForVTable);
}

llvm::GlobalVariable *
CodeGenVTables::GenerateConstructionVTable(const CXXRecordDecl *RD, 
                                      const BaseSubobject &Base, 
                                      bool BaseIsVirtual, 
                                   llvm::GlobalVariable::LinkageTypes Linkage,
                                      VTableAddressPointsMapTy& AddressPoints) {
  OwningPtr<VTableLayout> VTLayout(
    VTContext.createConstructionVTableLayout(Base.getBase(),
                                             Base.getBaseOffset(),
                                             BaseIsVirtual, RD));

  // Add the address points.
  AddressPoints = VTLayout->getAddressPoints();

  // Get the mangled construction vtable name.
  SmallString<256> OutName;
  llvm::raw_svector_ostream Out(OutName);
  CGM.getCXXABI().getMangleContext().
    mangleCXXCtorVTable(RD, Base.getBaseOffset().getQuantity(), Base.getBase(), 
                        Out);
  Out.flush();
  StringRef Name = OutName.str();

  llvm::ArrayType *ArrayType = 
    llvm::ArrayType::get(CGM.Int8PtrTy, VTLayout->getNumVTableComponents());

  // Create the variable that will hold the construction vtable.
  llvm::GlobalVariable *VTable = 
    CGM.CreateOrReplaceCXXRuntimeVariable(Name, ArrayType, Linkage);
  CGM.setTypeVisibility(VTable, RD, CodeGenModule::TVK_ForConstructionVTable);

  // V-tables are always unnamed_addr.
  VTable->setUnnamedAddr(true);

  // Create and set the initializer.
  llvm::Constant *Init = 
    CreateVTableInitializer(Base.getBase(), 
                            VTLayout->vtable_component_begin(), 
                            VTLayout->getNumVTableComponents(),
                            VTLayout->vtable_thunk_begin(),
                            VTLayout->getNumVTableThunks());
  VTable->setInitializer(Init);
  
  return VTable;
}

void 
CodeGenVTables::GenerateClassData(llvm::GlobalVariable::LinkageTypes Linkage,
                                  const CXXRecordDecl *RD) {
  llvm::GlobalVariable *VTable = GetAddrOfVTable(RD);
  if (VTable->hasInitializer())
    return;

  EmitVTableDefinition(VTable, Linkage, RD);

  if (RD->getNumVBases()) {
    llvm::GlobalVariable *VTT = GetAddrOfVTT(RD);
    EmitVTTDefinition(VTT, Linkage, RD);
  }

  // If this is the magic class __cxxabiv1::__fundamental_type_info,
  // we will emit the typeinfo for the fundamental types. This is the
  // same behaviour as GCC.
  const DeclContext *DC = RD->getDeclContext();
  if (RD->getIdentifier() &&
      RD->getIdentifier()->isStr("__fundamental_type_info") &&
      isa<NamespaceDecl>(DC) &&
      cast<NamespaceDecl>(DC)->getIdentifier() &&
      cast<NamespaceDecl>(DC)->getIdentifier()->isStr("__cxxabiv1") &&
      DC->getParent()->isTranslationUnit())
    CGM.EmitFundamentalRTTIDescriptors();
}
