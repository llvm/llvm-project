//===- WebAssembly.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ABIInfoImpl.h"
#include "TargetInfo.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/Intrinsics.h"

#include "clang/AST/ParentMapContext.h"

using namespace clang;
using namespace clang::CodeGen;

#define DEBUG_TYPE "clang-target-wasm"

//===----------------------------------------------------------------------===//
// WebAssembly ABI Implementation
//
// This is a very simple ABI that relies a lot on DefaultABIInfo.
//===----------------------------------------------------------------------===//

class WebAssemblyABIInfo final : public ABIInfo {
  DefaultABIInfo defaultInfo;
  WebAssemblyABIKind Kind;

public:
  explicit WebAssemblyABIInfo(CodeGen::CodeGenTypes &CGT,
                              WebAssemblyABIKind Kind)
      : ABIInfo(CGT), defaultInfo(CGT), Kind(Kind) {}

private:
  ABIArgInfo classifyReturnType(QualType RetTy) const;
  ABIArgInfo classifyArgumentType(QualType Ty) const;

  // DefaultABIInfo's classifyReturnType and classifyArgumentType are
  // non-virtual, but computeInfo and EmitVAArg are virtual, so we
  // overload them.
  void computeInfo(CGFunctionInfo &FI) const override {
    if (!getCXXABI().classifyReturnType(FI))
      FI.getReturnInfo() = classifyReturnType(FI.getReturnType());
    for (auto &Arg : FI.arguments())
      Arg.info = classifyArgumentType(Arg.type);
  }

  RValue EmitVAArg(CodeGenFunction &CGF, Address VAListAddr, QualType Ty,
                   AggValueSlot Slot) const override;
};

class WebAssemblyTargetCodeGenInfo final : public TargetCodeGenInfo {
public:
  explicit WebAssemblyTargetCodeGenInfo(CodeGen::CodeGenTypes &CGT,
                                        WebAssemblyABIKind K)
      : TargetCodeGenInfo(std::make_unique<WebAssemblyABIInfo>(CGT, K)) {
    SwiftInfo =
        std::make_unique<SwiftABIInfo>(CGT, /*SwiftErrorInRegister=*/false);
    ThunkCache = llvm::StringMap<llvm::Function *>();
  }

  void setTargetAttributes(const Decl *D, llvm::GlobalValue *GV,
                           CodeGen::CodeGenModule &CGM) const override {
    TargetCodeGenInfo::setTargetAttributes(D, GV, CGM);
    if (const auto *FD = dyn_cast_or_null<FunctionDecl>(D)) {
      if (const auto *Attr = FD->getAttr<WebAssemblyImportModuleAttr>()) {
        llvm::Function *Fn = cast<llvm::Function>(GV);
        llvm::AttrBuilder B(GV->getContext());
        B.addAttribute("wasm-import-module", Attr->getImportModule());
        Fn->addFnAttrs(B);
      }
      if (const auto *Attr = FD->getAttr<WebAssemblyImportNameAttr>()) {
        llvm::Function *Fn = cast<llvm::Function>(GV);
        llvm::AttrBuilder B(GV->getContext());
        B.addAttribute("wasm-import-name", Attr->getImportName());
        Fn->addFnAttrs(B);
      }
      if (const auto *Attr = FD->getAttr<WebAssemblyExportNameAttr>()) {
        llvm::Function *Fn = cast<llvm::Function>(GV);
        llvm::AttrBuilder B(GV->getContext());
        B.addAttribute("wasm-export-name", Attr->getExportName());
        Fn->addFnAttrs(B);
      }
    }

    if (auto *FD = dyn_cast_or_null<FunctionDecl>(D)) {
      llvm::Function *Fn = cast<llvm::Function>(GV);
      if (!FD->doesThisDeclarationHaveABody() && !FD->hasPrototype())
        Fn->addFnAttr("no-prototype");
    }
  }

  /// Return the WebAssembly externref reference type.
  virtual llvm::Type *getWasmExternrefReferenceType() const override {
    return llvm::Type::getWasm_ExternrefTy(getABIInfo().getVMContext());
  }
  /// Return the WebAssembly funcref reference type.
  virtual llvm::Type *getWasmFuncrefReferenceType() const override {
    return llvm::Type::getWasm_FuncrefTy(getABIInfo().getVMContext());
  }

  virtual const DeclRefExpr *
  getWasmFunctionDeclRefExpr(const Expr *E, ASTContext &Ctx) const override {
    // Go down in the tree until finding the DeclRefExpr
    const DeclRefExpr *DRE = findDeclRefExpr(E);
    if (!DRE)
      return nullptr;

    // Final case. The argument is a declared function
    if (isa<FunctionDecl>(DRE->getDecl())) {
      return DRE;
    }

    // Complex case. The argument is a variable, we need to check
    // every assignment of the variable and see if we are bitcasting
    // or not.
    if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
      DRE = findDeclRefExprForVarUp(E, VD, Ctx);
      if (DRE)
        return DRE;

      // If no assignment exists on every parent scope, check for the
      // initialization
      if (!DRE && VD->hasInit()) {
        return getWasmFunctionDeclRefExpr(VD->getInit(), Ctx);
      }
    }

    return nullptr;
  }

  virtual llvm::Function *getOrCreateWasmFunctionPointerThunk(
      CodeGenModule &CGM, llvm::Value *OriginalFnPtr, QualType SrcType,
      QualType DstType) const override {

    // Get the signatures
    const FunctionProtoType *SrcProtoType = SrcType->getAs<FunctionProtoType>();
    const FunctionProtoType *DstProtoType = DstType->getAs<PointerType>()
                                                ->getPointeeType()
                                                ->getAs<FunctionProtoType>();

    // Check parameter counts: source must have same or fewer params than destination
    unsigned SrcParams = SrcProtoType->getNumParams();
    unsigned DstParams = DstProtoType->getNumParams();

    if (SrcParams > DstParams)
      return nullptr;  // Can't remove parameters

    // Check return types: compare LLVM types, not C types.
    QualType SrcRetTy = SrcProtoType->getReturnType();
    QualType DstRetTy = DstProtoType->getReturnType();
    llvm::Type *SrcRetLLVMTy = CGM.getTypes().ConvertType(SrcRetTy);
    llvm::Type *DstRetLLVMTy = CGM.getTypes().ConvertType(DstRetTy);
    bool sameReturnType = SrcRetLLVMTy == DstRetLLVMTy;

    if (!DstProtoType->getReturnType()->isVoidType() && !sameReturnType)
      return nullptr;  // Can't invent return values

    // Reject if signatures are identical (no adaptation needed)
    if (SrcParams == DstParams && sameReturnType)
      return nullptr;

    // Get the llvm function types
    llvm::FunctionType *DstFunctionType = llvm::cast<llvm::FunctionType>(
        CGM.getTypes().ConvertType(QualType(DstProtoType, 0)));
    llvm::FunctionType *SrcFunctionType = llvm::cast<llvm::FunctionType>(
        CGM.getTypes().ConvertType(QualType(SrcProtoType, 0)));

    // Construct the Thunk function with the Target (destination) signature
    std::string ThunkName = getThunkName(OriginalFnPtr->getName().str(),
                                         DstProtoType, CGM.getContext());
    // Check if we already have a thunk for this function
    if (auto It = ThunkCache.find(ThunkName); It != ThunkCache.end()) {
      LLVM_DEBUG(llvm::dbgs() << "getOrCreateWasmFunctionPointerThunk: "
                              << "found existing thunk for "
                              << OriginalFnPtr->getName().str() << " as "
                              << ThunkName << "\n");
      return It->second;
    }

    // Create the thunk function
    llvm::Module &M = CGM.getModule();
    llvm::Function *Thunk = llvm::Function::Create(
        DstFunctionType, llvm::Function::InternalLinkage, ThunkName, M);

    // Build the thunk body
    llvm::IRBuilder<> Builder(
        llvm::BasicBlock::Create(M.getContext(), "entry", Thunk));

    // Gather the arguments for calling the original function
    std::vector<llvm::Value *> CallArgs;
    unsigned CallN = SrcProtoType->getNumParams();

    auto ArgIt = Thunk->arg_begin();
    for (unsigned i = 0; i < CallN && ArgIt != Thunk->arg_end(); ++i, ++ArgIt) {
      llvm::Value *A = &*ArgIt;
      CallArgs.push_back(A);
    }

    // Create the call to the original function pointer
    llvm::CallInst *Call =
        Builder.CreateCall(SrcFunctionType, OriginalFnPtr, CallArgs);

    // Handle return type
    llvm::Type *ThunkRetTy = DstFunctionType->getReturnType();

    if (ThunkRetTy->isVoidTy()) {
      Builder.CreateRetVoid();
    } else {
      llvm::Value *Ret = Call;
      if (Ret->getType() != ThunkRetTy)
        Ret = Builder.CreateBitOrPointerCast(Ret, ThunkRetTy);
      Builder.CreateRet(Ret);
    }
    LLVM_DEBUG(llvm::dbgs() << "getOrCreateWasmFunctionPointerThunk:"
                            << " from " << OriginalFnPtr->getName().str()
                            << " to " << ThunkName << "\n");
    // Cache the thunk
    ThunkCache[ThunkName] = Thunk;
    return Thunk;
  }

  llvm::Value *emitWasmRuntimeFunctionPointerBinding(
      CodeGenFunction &CGF, llvm::Value *FnPtr, QualType SrcType,
      QualType DstType, bool IsImmediate) const override;

private:
  // The thunk cache for compile-time thunks
  mutable llvm::StringMap<llvm::Function *> ThunkCache;

  // Runtime thunk cache: maps (SrcSig, DstSig) -> wrapper function
  // The wrapper takes a function pointer and returns a thunk for it
  mutable llvm::DenseMap<std::pair<const FunctionProtoType*, const FunctionProtoType*>,
                         llvm::Function*> RuntimeWrapperCache;

  // Build the thunk name: "%s_{OrigName}_{WasmSig}"
  std::string getThunkName(std::string OrigName,
                           const FunctionProtoType *DstProto,
                           const ASTContext &Ctx) const;
  std::string getRuntimeWrapperName(const FunctionProtoType *SrcProto,
                                   const FunctionProtoType *DstProto,
                                   const ASTContext &Ctx) const;
  char getTypeSig(const QualType &Ty, const ASTContext &Ctx) const;
  std::string sanitizeTypeString(const std::string &typeStr) const;
  std::string getTypeName(const QualType &qt, const ASTContext &Ctx) const;
  const DeclRefExpr *findDeclRefExpr(const Expr *E) const;
  const DeclRefExpr *findDeclRefExprForVarDown(const Stmt *Parent,
                                               const VarDecl *V,
                                               ASTContext &Ctx) const;
  const DeclRefExpr *findDeclRefExprForVarUp(const Expr *E, const VarDecl *V,
                                             ASTContext &Ctx) const;
};

/// Classify argument of given type \p Ty.
ABIArgInfo WebAssemblyABIInfo::classifyArgumentType(QualType Ty) const {
  Ty = useFirstFieldIfTransparentUnion(Ty);

  if (isAggregateTypeForABI(Ty)) {
    // Records with non-trivial destructors/copy-constructors should not be
    // passed by value.
    if (auto RAA = getRecordArgABI(Ty, getCXXABI()))
      return getNaturalAlignIndirect(Ty, getDataLayout().getAllocaAddrSpace(),
                                     RAA == CGCXXABI::RAA_DirectInMemory);
    // Ignore empty structs/unions.
    if (isEmptyRecord(getContext(), Ty, true))
      return ABIArgInfo::getIgnore();
    // Lower single-element structs to just pass a regular value. TODO: We
    // could do reasonable-size multiple-element structs too, using getExpand(),
    // though watch out for things like bitfields.
    if (const Type *SeltTy = isSingleElementStruct(Ty, getContext()))
      return ABIArgInfo::getDirect(CGT.ConvertType(QualType(SeltTy, 0)));
    // For the experimental multivalue ABI, fully expand all other aggregates
    if (Kind == WebAssemblyABIKind::ExperimentalMV) {
      if (Ty->getAs<ComplexType>())
        return ABIArgInfo::getDirect();
      const auto *RD = Ty->getAsRecordDecl();
      if (RD) {
        bool HasBitField = false;
        for (auto *Field : RD->fields()) {
          if (Field->isBitField()) {
            HasBitField = true;
            break;
          }
        }
        if (!HasBitField)
          return ABIArgInfo::getExpand();
      }
    }
  }

  // Otherwise just do the default thing.
  return defaultInfo.classifyArgumentType(Ty);
}

ABIArgInfo WebAssemblyABIInfo::classifyReturnType(QualType RetTy) const {
  if (isAggregateTypeForABI(RetTy)) {
    // Records with non-trivial destructors/copy-constructors should not be
    // returned by value.
    if (!getRecordArgABI(RetTy, getCXXABI())) {
      // Ignore empty structs/unions.
      if (isEmptyRecord(getContext(), RetTy, true))
        return ABIArgInfo::getIgnore();
      // Lower single-element structs to just return a regular value. TODO: We
      // could do reasonable-size multiple-element structs too, using
      // ABIArgInfo::getDirect().
      if (const Type *SeltTy = isSingleElementStruct(RetTy, getContext()))
        return ABIArgInfo::getDirect(CGT.ConvertType(QualType(SeltTy, 0)));
      // For the experimental multivalue ABI, return all other aggregates
      if (Kind == WebAssemblyABIKind::ExperimentalMV)
        return ABIArgInfo::getDirect();
    }
  }

  // Otherwise just do the default thing.
  return defaultInfo.classifyReturnType(RetTy);
}

RValue WebAssemblyABIInfo::EmitVAArg(CodeGenFunction &CGF, Address VAListAddr,
                                     QualType Ty, AggValueSlot Slot) const {
  bool IsIndirect = isAggregateTypeForABI(Ty) &&
                    !isEmptyRecord(getContext(), Ty, true) &&
                    !isSingleElementStruct(Ty, getContext());
  return emitVoidPtrVAArg(CGF, VAListAddr, Ty, IsIndirect,
                          getContext().getTypeInfoInChars(Ty),
                          CharUnits::fromQuantity(4),
                          /*AllowHigherAlign=*/true, Slot);
}

// Generate wrapper name for runtime function pointer binding
std::string WebAssemblyTargetCodeGenInfo::getRuntimeWrapperName(
    const FunctionProtoType *SrcProto, const FunctionProtoType *DstProto,
    const ASTContext &Ctx) const {
  std::string Name = "__wasm_runtime_wrapper_";

  // Encode source signature
  QualType SrcRetTy = SrcProto->getReturnType();
  if (SrcRetTy->isVoidType()) {
    Name += 'v';
  } else {
    Name += getTypeSig(SrcRetTy, Ctx);
  }
  for (QualType ParamType : SrcProto->param_types()) {
    Name += getTypeSig(ParamType, Ctx);
  }

  Name += "_to_";

  // Encode destination signature
  QualType DstRetTy = DstProto->getReturnType();
  if (DstRetTy->isVoidType()) {
    Name += 'v';
  } else {
    Name += getTypeSig(DstRetTy, Ctx);
  }
  for (QualType ParamType : DstProto->param_types()) {
    Name += getTypeSig(ParamType, Ctx);
  }

  return Name;
}

// Emit runtime binding for function pointer cast
// This handles cases like g_list_free_full where a runtime parameter
// needs to be cast from fewer params to more params
llvm::Value *WebAssemblyTargetCodeGenInfo::emitWasmRuntimeFunctionPointerBinding(
    CodeGenFunction &CGF, llvm::Value *FnPtr, QualType SrcType,
    QualType DstType, bool IsImmediate) const {

  const FunctionProtoType *SrcProto =
      SrcType->getPointeeType()->getAs<FunctionProtoType>();
  const FunctionProtoType *DstProto =
      DstType->getPointeeType()->getAs<FunctionProtoType>();

  if (!SrcProto || !DstProto)
    return nullptr;

  // Check parameter counts: source must have same or fewer params than destination
  // We can add parameters (caller provides them, we ignore extras when calling source)
  // We cannot remove parameters (caller doesn't provide them, we can't invent values)
  unsigned SrcParams = SrcProto->getNumParams();
  unsigned DstParams = DstProto->getNumParams();

  if (SrcParams > DstParams)
    return nullptr;  // Can't remove parameters

  // Check return types: we can discard a return value but cannot invent one.
  // Compare LLVM types (not C types) since wasm only cares about i32/i64/f32/f64.
  QualType SrcRetTy = SrcProto->getReturnType();
  QualType DstRetTy = DstProto->getReturnType();
  llvm::Type *SrcRetLLVMTy = CGF.CGM.getTypes().ConvertType(SrcRetTy);
  llvm::Type *DstRetLLVMTy = CGF.CGM.getTypes().ConvertType(DstRetTy);
  bool sameReturnType = SrcRetLLVMTy == DstRetLLVMTy;

  if (!DstRetTy->isVoidType() && !sameReturnType)
    return nullptr;  // Can't invent return values

  // Reject if signatures are identical (no adaptation needed)
  if (SrcParams == DstParams && sameReturnType)
    return nullptr;

  // A null function pointer needs no wrapper — fall through to bitcast
  if (isa<llvm::ConstantPointerNull>(FnPtr))
    return nullptr;

  LLVM_DEBUG(llvm::dbgs() << "emitWasmRuntimeFunctionPointerBinding: "
                          << "src params=" << SrcParams
                          << " dst params=" << DstParams << "\n");

  llvm::Module &M = CGF.CGM.getModule();
  llvm::LLVMContext &Context = M.getContext();
  llvm::PointerType *PtrTy = llvm::PointerType::getUnqual(Context);
  llvm::Type *I32Ty = llvm::IntegerType::getInt32Ty(Context);

  // Pre-allocated pool: N wrapper functions + N TLS slots per signature pair.
  // Each runtime invocation atomically claims a slot. This supports both
  // "call immediately" and "store for later" patterns without overwrites.
  static const unsigned POOL_SIZE = 64;

  std::string WrapperName = getRuntimeWrapperName(SrcProto, DstProto, CGF.CGM.getContext());

  std::string SourceId = M.getSourceFileName();
  if (SourceId.empty())
    SourceId = M.getName();
  for (char &C : SourceId)
    if (!isalnum(C) && C != '_')
      C = '_';
  WrapperName += "_" + SourceId;

  std::string PoolName = "__wasm_runtime_pool_" + WrapperName;

  // Get or create pool globals (once per module per signature pair)
  llvm::GlobalVariable *Counter = M.getNamedGlobal(PoolName + "_counter");
  llvm::GlobalVariable *ImmediateSlot = M.getNamedGlobal(PoolName + "_immediate_slot");
  llvm::Function *ImmediateWrapper = M.getFunction(WrapperName + "_immediate");
  llvm::ArrayType *SlotArrayTy = llvm::ArrayType::get(PtrTy, POOL_SIZE);
  llvm::GlobalVariable *Slots = nullptr;
  llvm::GlobalVariable *WrapperTable = nullptr;
  llvm::FunctionType *SrcFnType = nullptr;
  llvm::FunctionType *DstFnType = nullptr;

  if (!Counter) {
    SrcFnType = llvm::cast<llvm::FunctionType>(
        CGF.CGM.getTypes().ConvertType(QualType(SrcProto, 0)));
    DstFnType = llvm::cast<llvm::FunctionType>(
        CGF.CGM.getTypes().ConvertType(QualType(DstProto, 0)));

    Counter = new llvm::GlobalVariable(
        M, I32Ty, false, llvm::GlobalValue::InternalLinkage,
        llvm::ConstantInt::get(I32Ty, 0), PoolName + "_counter");
    Counter->setThreadLocalMode(llvm::GlobalValue::GeneralDynamicTLSModel);

    // Immediate-call TLS slot: per-thread, no races, reused every call
    ImmediateSlot = new llvm::GlobalVariable(
        M, PtrTy, false, llvm::GlobalValue::InternalLinkage,
        llvm::ConstantPointerNull::get(PtrTy), PoolName + "_immediate_slot");
    ImmediateSlot->setThreadLocalMode(llvm::GlobalValue::GeneralDynamicTLSModel);

    // Immediate wrapper: loads from TLS slot, calls with adapted signature
    ImmediateWrapper = llvm::Function::Create(
        DstFnType, llvm::GlobalValue::InternalLinkage,
        WrapperName + "_immediate", M);
    ImmediateWrapper->addFnAttr(llvm::Attribute::NoInline);
    ImmediateWrapper->addFnAttr(llvm::Attribute::NoUnwind);
    {
      llvm::BasicBlock *BB = llvm::BasicBlock::Create(Context, "entry", ImmediateWrapper);
      llvm::IRBuilder<> B(BB);
      llvm::Value *FP = B.CreateLoad(PtrTy, ImmediateSlot);
      llvm::BasicBlock *CallBB = llvm::BasicBlock::Create(Context, "call", ImmediateWrapper);
      llvm::BasicBlock *NullBB = llvm::BasicBlock::Create(Context, "nullslot", ImmediateWrapper);
      B.CreateCondBr(B.CreateIsNotNull(FP), CallBB, NullBB);
      B.SetInsertPoint(CallBB);
      llvm::SmallVector<llvm::Value *, 8> ImmArgs;
      auto AI = ImmediateWrapper->arg_begin();
      for (unsigned J = 0; J < SrcParams && AI != ImmediateWrapper->arg_end(); ++J, ++AI) {
        llvm::Value *A = &*AI;
        if (A->getType() != SrcFnType->getParamType(J))
          A = B.CreateBitOrPointerCast(A, SrcFnType->getParamType(J));
        ImmArgs.push_back(A);
      }
      llvm::CallInst *ImmCall = B.CreateCall(SrcFnType, FP, ImmArgs);
      if (DstFnType->getReturnType()->isVoidTy()) {
        B.CreateRetVoid(); B.SetInsertPoint(NullBB); B.CreateRetVoid();
      } else {
        llvm::Value *R = ImmCall;
        if (R->getType() != DstFnType->getReturnType())
          R = B.CreateBitOrPointerCast(R, DstFnType->getReturnType());
        B.CreateRet(R);
        B.SetInsertPoint(NullBB);
        B.CreateRet(llvm::Constant::getNullValue(DstFnType->getReturnType()));
      }
    }

    Slots = new llvm::GlobalVariable(
        M, SlotArrayTy, false, llvm::GlobalValue::InternalLinkage,
        llvm::ConstantAggregateZero::get(SlotArrayTy), PoolName + "_slots");

    // 8-entry direct-mapped cache: avoids pool allocation for repeated fn_ptrs
    static const unsigned CACHE_SIZE = 8;
    llvm::ArrayType *CacheTy = llvm::ArrayType::get(PtrTy, CACHE_SIZE);
    new llvm::GlobalVariable(
        M, CacheTy, false, llvm::GlobalValue::InternalLinkage,
        llvm::ConstantAggregateZero::get(CacheTy), PoolName + "_cache_keys");
    new llvm::GlobalVariable(
        M, CacheTy, false, llvm::GlobalValue::InternalLinkage,
        llvm::ConstantAggregateZero::get(CacheTy), PoolName + "_cache_wrappers");

    // Pre-generate POOL_SIZE wrapper functions + build lookup table
    llvm::SmallVector<llvm::Constant *, 64> WrappersConst;
    for (unsigned I = 0; I < POOL_SIZE; ++I) {
      std::string InstName = WrapperName + "_" + std::to_string(I);
      llvm::Function *W = llvm::Function::Create(
          DstFnType, llvm::GlobalValue::InternalLinkage, InstName, M);
      W->addFnAttr(llvm::Attribute::NoInline);
      W->addFnAttr(llvm::Attribute::NoUnwind);

      llvm::BasicBlock *BB = llvm::BasicBlock::Create(Context, "entry", W);
      llvm::IRBuilder<> B(BB);

      llvm::Value *SlotPtr = B.CreateConstInBoundsGEP1_32(PtrTy, Slots, I);
      llvm::Value *FP = B.CreateLoad(PtrTy, SlotPtr);

      // Defensive null check: if slot was never written, skip call
      llvm::BasicBlock *CallBB = llvm::BasicBlock::Create(Context, "call", W);
      llvm::BasicBlock *NullBB = llvm::BasicBlock::Create(Context, "nullslot", W);
      llvm::Value *IsNotNull = B.CreateIsNotNull(FP);
      B.CreateCondBr(IsNotNull, CallBB, NullBB);

      B.SetInsertPoint(CallBB);
      llvm::SmallVector<llvm::Value *, 8> CallArgs;
      auto ArgIt = W->arg_begin();
      for (unsigned J = 0; J < SrcParams && ArgIt != W->arg_end(); ++J, ++ArgIt) {
        llvm::Value *A = &*ArgIt;
        if (A->getType() != SrcFnType->getParamType(J))
          A = B.CreateBitOrPointerCast(A, SrcFnType->getParamType(J));
        CallArgs.push_back(A);
      }
      llvm::CallInst *Call = B.CreateCall(SrcFnType, FP, CallArgs);

      if (DstFnType->getReturnType()->isVoidTy()) {
        B.CreateRetVoid();
        B.SetInsertPoint(NullBB);
        B.CreateRetVoid();
      } else {
        llvm::Value *Ret = Call;
        if (Ret->getType() != DstFnType->getReturnType())
          Ret = B.CreateBitOrPointerCast(Ret, DstFnType->getReturnType());
        B.CreateRet(Ret);
        B.SetInsertPoint(NullBB);
        B.CreateRet(llvm::Constant::getNullValue(DstFnType->getReturnType()));
      }

      WrappersConst.push_back(llvm::ConstantExpr::getBitCast(W, PtrTy));
    }

    // Create constant lookup table (not TLS — read-only function pointers)
    llvm::ArrayType *WrapTblTy = llvm::ArrayType::get(PtrTy, POOL_SIZE);
    WrapperTable = new llvm::GlobalVariable(
        M, WrapTblTy, true, llvm::GlobalValue::InternalLinkage,
        llvm::ConstantArray::get(WrapTblTy, WrappersConst),
        PoolName + "_wrappers");
    WrapperTable->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
  } else {
    Slots = M.getNamedGlobal(PoolName + "_slots");
    WrapperTable = M.getNamedGlobal(PoolName + "_wrappers");
    SrcFnType = llvm::cast<llvm::FunctionType>(
        CGF.CGM.getTypes().ConvertType(QualType(SrcProto, 0)));
  }

  // Runtime null check
  llvm::Value *IsNull = CGF.Builder.CreateIsNull(FnPtr);
  llvm::BasicBlock *NullContBB = llvm::BasicBlock::Create(Context, "nullcont", CGF.CurFn);
  llvm::BasicBlock *NotNullBB = llvm::BasicBlock::Create(Context, "notnull", CGF.CurFn);
  CharUnits PtrAlign = CGF.CGM.getPointerAlign();

  if (IsImmediate) {
    // === Immediate call: 1 TLS slot + 1 wrapper, no branches after null check ===
    CGF.Builder.CreateCondBr(IsNull, NullContBB, NotNullBB);

    CGF.Builder.SetInsertPoint(NotNullBB);
    CGF.Builder.CreateStore(FnPtr, Address(ImmediateSlot, PtrTy, PtrAlign));
    CGF.Builder.CreateBr(NullContBB);

    CGF.Builder.SetInsertPoint(NullContBB);
    llvm::PHINode *PHI = CGF.Builder.CreatePHI(PtrTy, 2);
    PHI->addIncoming(llvm::ConstantExpr::getBitCast(ImmediateWrapper, PtrTy), NotNullBB);
    PHI->addIncoming(llvm::ConstantPointerNull::get(PtrTy), NullContBB);
    return PHI;
  }

  // === Store-for-later: pool with 64 slots + 8-entry cache + atomic counter ===
  CGF.Builder.CreateCondBr(IsNull, NullContBB, NotNullBB);

  CGF.Builder.SetInsertPoint(NotNullBB);
  llvm::ArrayType *WrapTblTy = llvm::ArrayType::get(PtrTy, POOL_SIZE);
  static const unsigned CACHE_SIZE = 8;
  llvm::ArrayType *CacheTy = llvm::ArrayType::get(PtrTy, CACHE_SIZE);
  llvm::GlobalVariable *CacheKeys = M.getNamedGlobal(PoolName + "_cache_keys");
  llvm::GlobalVariable *CacheWrappers = M.getNamedGlobal(PoolName + "_cache_wrappers");
  llvm::BasicBlock *CacheHitBB = llvm::BasicBlock::Create(Context, "cachehit", CGF.CurFn);
  llvm::BasicBlock *CacheMissBB = llvm::BasicBlock::Create(Context, "cachemiss", CGF.CurFn);
  llvm::BasicBlock *ScanBB = llvm::BasicBlock::Create(Context, "scan", CGF.CurFn);
  llvm::BasicBlock *ScanFoundBB = llvm::BasicBlock::Create(Context, "scanfnd", CGF.CurFn);
  llvm::BasicBlock *ScanNextBB = llvm::BasicBlock::Create(Context, "scannxt", CGF.CurFn);
  llvm::BasicBlock *AllocCheckBB = llvm::BasicBlock::Create(Context, "allocchk", CGF.CurFn);
  llvm::BasicBlock *AllocStoreBB = llvm::BasicBlock::Create(Context, "allocstr", CGF.CurFn);
  llvm::BasicBlock *OverflowBB = llvm::BasicBlock::Create(Context, "overflow", CGF.CurFn);
  llvm::BasicBlock *ContBB = llvm::BasicBlock::Create(Context, "cont", CGF.CurFn);

  // Cache lookup: idx = (fn_ptr >> 2) & 7
  llvm::Value *CacheIdx = CGF.Builder.CreateAnd(
      CGF.Builder.CreateLShr(
          CGF.Builder.CreatePtrToInt(FnPtr, I32Ty),
          llvm::ConstantInt::get(I32Ty, 2)),
      llvm::ConstantInt::get(I32Ty, CACHE_SIZE - 1));
  llvm::Value *CacheKeyGEP = CGF.Builder.CreateInBoundsGEP(
      CacheTy, CacheKeys, {llvm::ConstantInt::get(I32Ty, 0), CacheIdx});
  Address CacheKeyAddr(CacheKeyGEP, PtrTy, PtrAlign);
  llvm::Value *CachedFn = CGF.Builder.CreateLoad(CacheKeyAddr);
  llvm::Value *CacheHit = CGF.Builder.CreateICmpEQ(CachedFn, FnPtr);
  llvm::Value *CacheWrapGEP = CGF.Builder.CreateInBoundsGEP(
      CacheTy, CacheWrappers, {llvm::ConstantInt::get(I32Ty, 0), CacheIdx});
  Address CacheWrapAddr(CacheWrapGEP, PtrTy, PtrAlign);

  CGF.Builder.CreateCondBr(CacheHit, CacheHitBB, CacheMissBB);

  // Cache hit: load cached wrapper, branch to cont
  CGF.Builder.SetInsertPoint(CacheHitBB);
  llvm::Value *CacheHitW = CGF.Builder.CreateLoad(CacheWrapAddr);
  CGF.Builder.CreateBr(ContBB);

  // Cache miss: scan pool for existing mapping
  CGF.Builder.SetInsertPoint(CacheMissBB);
  CGF.Builder.CreateBr(ScanBB);

  // Scan loop: find fn_ptr in slots[0..POOL_SIZE-1]
  CGF.Builder.SetInsertPoint(ScanBB);
  llvm::PHINode *ScanIdx = CGF.Builder.CreatePHI(I32Ty, 2);
  ScanIdx->addIncoming(llvm::ConstantInt::get(I32Ty, 0), CacheMissBB);
  llvm::Value *ScanSlotGEP = CGF.Builder.CreateInBoundsGEP(
      SlotArrayTy, Slots, {llvm::ConstantInt::get(I32Ty, 0), ScanIdx});
  llvm::Value *ScanFn = CGF.Builder.CreateLoad(Address(ScanSlotGEP, PtrTy, PtrAlign));
  llvm::Value *ScanMatch = CGF.Builder.CreateICmpEQ(ScanFn, FnPtr);
  CGF.Builder.CreateCondBr(ScanMatch, ScanFoundBB, ScanNextBB);

  // Found existing slot: update cache, return existing wrapper
  CGF.Builder.SetInsertPoint(ScanFoundBB);
  CGF.Builder.CreateStore(FnPtr, CacheKeyAddr);
  llvm::Value *FoundWrapGEP = CGF.Builder.CreateInBoundsGEP(
      WrapTblTy, WrapperTable, {llvm::ConstantInt::get(I32Ty, 0), ScanIdx});
  llvm::Value *FoundW = CGF.Builder.CreateLoad(Address(FoundWrapGEP, PtrTy, PtrAlign));
  CGF.Builder.CreateStore(FoundW, CacheWrapAddr);
  CGF.Builder.CreateBr(ContBB);

  // Advance scan
  CGF.Builder.SetInsertPoint(ScanNextBB);
  llvm::Value *NextIdx = CGF.Builder.CreateAdd(
      ScanIdx, llvm::ConstantInt::get(I32Ty, 1));
  llvm::Value *ScanEnd = CGF.Builder.CreateICmpUGE(
      NextIdx, llvm::ConstantInt::get(I32Ty, POOL_SIZE));
  ScanIdx->addIncoming(NextIdx, ScanNextBB);
  CGF.Builder.CreateCondBr(ScanEnd, AllocCheckBB, ScanBB);

  // Allocate new slot: atomic counter increment
  CGF.Builder.SetInsertPoint(AllocCheckBB);
  Address CounterAddr(Counter, I32Ty, PtrAlign);
  llvm::Value *Slot = CGF.Builder.CreateAtomicRMW(
      llvm::AtomicRMWInst::Add, CounterAddr,
      llvm::ConstantInt::get(I32Ty, 1), llvm::AtomicOrdering::Monotonic);
  llvm::Value *InBounds = CGF.Builder.CreateICmpULT(
      Slot, llvm::ConstantInt::get(I32Ty, POOL_SIZE));
  CGF.Builder.CreateCondBr(InBounds, AllocStoreBB, OverflowBB);

  // Overflow
  CGF.Builder.SetInsertPoint(OverflowBB);
  CGF.Builder.CreateCall(llvm::Intrinsic::getOrInsertDeclaration(
      &M, llvm::Intrinsic::trap));
  CGF.Builder.CreateUnreachable();

  // Store in pool + update cache
  CGF.Builder.SetInsertPoint(AllocStoreBB);
  llvm::Value *SlotIdx[] = {llvm::ConstantInt::get(I32Ty, 0), Slot};
  llvm::Value *SlotGEP = CGF.Builder.CreateInBoundsGEP(
      SlotArrayTy, Slots, SlotIdx);
  CGF.Builder.CreateStore(FnPtr, Address(SlotGEP, PtrTy, PtrAlign));
  llvm::Value *WrapGEP = CGF.Builder.CreateInBoundsGEP(
      WrapTblTy, WrapperTable, SlotIdx);
  llvm::Value *W = CGF.Builder.CreateLoad(Address(WrapGEP, PtrTy, PtrAlign));
  CGF.Builder.CreateStore(FnPtr, CacheKeyAddr);
  CGF.Builder.CreateStore(W, CacheWrapAddr);
  CGF.Builder.CreateBr(ContBB);

  // Null path
  CGF.Builder.SetInsertPoint(NullContBB);
  CGF.Builder.CreateBr(ContBB);

  // ContBB: PHI for result
  CGF.Builder.SetInsertPoint(ContBB);
  llvm::PHINode *PHI = CGF.Builder.CreatePHI(PtrTy, 4);
  PHI->addIncoming(CacheHitW, CacheHitBB);
  PHI->addIncoming(FoundW, ScanFoundBB);
  PHI->addIncoming(W, AllocStoreBB);
  PHI->addIncoming(llvm::ConstantPointerNull::get(PtrTy), NullContBB);
  return PHI;
}

std::unique_ptr<TargetCodeGenInfo>
CodeGen::createWebAssemblyTargetCodeGenInfo(CodeGenModule &CGM,
                                            WebAssemblyABIKind K) {
  return std::make_unique<WebAssemblyTargetCodeGenInfo>(CGM.getTypes(), K);
}

// Helper to get the type signature character for a given QualType
// Returns a character that represents the given QualType in a wasm signature.
// See getInvokeSig() in WebAssemblyAsmPrinter for related logic.
char WebAssemblyTargetCodeGenInfo::getTypeSig(const QualType &Ty,
                                              const ASTContext &Ctx) const {
  if (Ty->isAnyPointerType()) {
    return Ctx.getTypeSize(Ctx.VoidPtrTy) == 32 ? 'i' : 'j';
  }
  if (Ty->isIntegerType()) {
    return Ctx.getTypeSize(Ty) <= 32 ? 'i' : 'j';
  }
  if (Ty->isFloatingType()) {
    return Ctx.getTypeSize(Ty) <= 32 ? 'f' : 'd';
  }
  if (Ty->isVectorType()) {
    return 'V';
  }
  if (Ty->isWebAssemblyTableType()) {
    return 'F';
  }
  if (Ty->isWebAssemblyExternrefType()) {
    return 'X';
  }

  llvm_unreachable("Unhandled QualType");
}

std::string
WebAssemblyTargetCodeGenInfo::getThunkName(std::string OrigName,
                                           const FunctionProtoType *DstProto,
                                           const ASTContext &Ctx) const {

  std::string ThunkName = "__" + OrigName + "_";
  QualType RetTy = DstProto->getReturnType();
  if (RetTy->isVoidType()) {
    ThunkName += 'v';
  } else {
    ThunkName += getTypeSig(RetTy, Ctx);
  }
  for (unsigned i = 0; i < DstProto->getNumParams(); ++i) {
    ThunkName += getTypeSig(DstProto->getParamType(i), Ctx);
  }
  return ThunkName;
}

/// Recursively find the first DeclRefExpr in an Expr subtree.
/// Returns nullptr if not found.
const DeclRefExpr *
WebAssemblyTargetCodeGenInfo::findDeclRefExpr(const Expr *E) const {
  if (!E)
    return nullptr;

  // In case it is a function call, abort
  if (isa<CallExpr>(E))
    return nullptr;

  // If this node is a DeclRefExpr, return it.
  if (const auto *DRE = dyn_cast<DeclRefExpr>(E))
    return DRE;

  // Otherwise, recurse into children.
  for (const Stmt *Child : E->children()) {
    if (const auto *ChildExpr = dyn_cast_or_null<Expr>(Child)) {
      if (const DeclRefExpr *Found = findDeclRefExpr(ChildExpr))
        return Found;
    }
  }
  return nullptr;
}

const DeclRefExpr *WebAssemblyTargetCodeGenInfo::findDeclRefExprForVarDown(
    const Stmt *Parent, const VarDecl *V, ASTContext &Ctx) const {
  if (!Parent)
    return nullptr;

  // Find down every assignment of V.
  // Standalone expression statements appear as Expr nodes directly under the
  // CompoundStmt, so cast each child to Expr (if possible) and check for
  // a BinaryOperator assignment.
  const BinaryOperator *A = nullptr;
  for (const Stmt *Child : Parent->children()) {
    const auto *BO = dyn_cast_or_null<BinaryOperator>(Child);
    if (!BO)
      if (const auto *E = dyn_cast_or_null<Expr>(Child))
        BO = dyn_cast<BinaryOperator>(E->IgnoreParenCasts());
    if (!BO || !BO->isAssignmentOp())
      continue;
    auto *LHS = llvm::dyn_cast<DeclRefExpr>(BO->getLHS());
    if (LHS && LHS->getDecl() == V)
      A = BO;
  }

  // We have an assignment of the Var, recurse in it
  if (A) {
    return getWasmFunctionDeclRefExpr(A->getRHS(), Ctx);
  }

  return nullptr;
}

const DeclRefExpr *WebAssemblyTargetCodeGenInfo::findDeclRefExprForVarUp(
    const Expr *E, const VarDecl *V, ASTContext &Ctx) const {
  // Use a DynTypedNode to walk parents, since a Stmt may be parented by a Decl
  // (e.g. a VarDecl initializer) and get<Stmt>() would return nullptr there.
  auto cur = clang::DynTypedNode::create(*E);
  while (true) {
    auto parents = Ctx.getParentMapContext().getParents(cur);
    if (parents.empty())
      break;
    cur = parents[0];
    if (const auto *CS = cur.get<clang::CompoundStmt>()) {
      const DeclRefExpr *DRE = findDeclRefExprForVarDown(CS, V, Ctx);
      if (DRE)
        return DRE;
    }
  }
  return nullptr;
}
