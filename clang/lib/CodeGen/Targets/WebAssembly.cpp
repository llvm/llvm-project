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

    // Check return types: we can discard a return value but cannot invent one
    QualType SrcRetTy = SrcProtoType->getReturnType();
    QualType DstRetTy = DstProtoType->getReturnType();
    bool sameReturnType = CGM.getContext().hasSameType(SrcRetTy, DstRetTy);

    if (!DstRetTy->isVoidType() && !sameReturnType)
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
      QualType DstType) const override;

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
      const auto *RD = Ty->castAsRecordDecl();
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
    QualType DstType) const {

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

  // Check return types: we can discard a return value but cannot invent one
  // Allow: int -> void (discard return), int -> int (pass through), void -> void
  // Reject: void -> int (can't invent return value)
  QualType SrcRetTy = SrcProto->getReturnType();
  QualType DstRetTy = DstProto->getReturnType();
  bool sameReturnType = CGF.getContext().hasSameType(SrcRetTy, DstRetTy);

  if (!DstRetTy->isVoidType() && !sameReturnType)
    return nullptr;  // Can't invent return values

  // Reject if signatures are identical (no adaptation needed)
  if (SrcParams == DstParams && sameReturnType)
    return nullptr;

  LLVM_DEBUG(llvm::dbgs() << "emitWasmRuntimeFunctionPointerBinding: "
                          << "src params=" << SrcParams
                          << " dst params=" << DstParams << "\n");

  auto Key = std::make_pair(SrcProto, DstProto);
  auto It = RuntimeWrapperCache.find(Key);

  llvm::Module &M = CGF.CGM.getModule();
  llvm::LLVMContext &Context = M.getContext();
  llvm::Type *PtrTy = llvm::PointerType::getUnqual(Context);

  std::string WrapperName = getRuntimeWrapperName(SrcProto, DstProto, CGF.CGM.getContext());
  std::string GlobalName = WrapperName + "_fptr";

  // Get or create the global variable for storing the function pointer
  // Use LinkOnceODRLinkage to match the wrapper function, allowing the linker
  // to merge globals across translation units into a single shared variable
  llvm::GlobalVariable *FnPtrGlobal = M.getNamedGlobal(GlobalName);
  if (!FnPtrGlobal) {
    FnPtrGlobal = new llvm::GlobalVariable(
        M, PtrTy, /*isConstant=*/false, llvm::GlobalValue::LinkOnceODRLinkage,
        llvm::Constant::getNullValue(PtrTy), GlobalName);
    // Make it thread-local to support WebAssembly threads
    FnPtrGlobal->setThreadLocalMode(llvm::GlobalValue::GeneralDynamicTLSModel);
  }

  llvm::Function *Wrapper;
  if (It != RuntimeWrapperCache.end()) {
    Wrapper = It->second;
  } else {
    // Create a new wrapper function that takes a function pointer
    // and returns a thunk with the destination signature

    llvm::FunctionType *SrcFnType = llvm::cast<llvm::FunctionType>(
        CGF.CGM.getTypes().ConvertType(QualType(SrcProto, 0)));
    llvm::FunctionType *DstFnType = llvm::cast<llvm::FunctionType>(
        CGF.CGM.getTypes().ConvertType(QualType(DstProto, 0)));

    // Wrapper signature: takes src function pointer, has dst signature
    // Use LinkOnceODRLinkage to:
    // 1. Prevent dead argument elimination (optimizer can't see all callers)
    // 2. Allow linker to merge duplicates across modules (no symbol collisions)
    // 3. Preserve exact signature required by WebAssembly type checking
    Wrapper = llvm::Function::Create(
        DstFnType, llvm::GlobalValue::LinkOnceODRLinkage, WrapperName, M);

    // Mark as noinline to prevent inlining that would expose unused parameters
    Wrapper->addFnAttr(llvm::Attribute::NoInline);
    Wrapper->addFnAttr(llvm::Attribute::NoUnwind);

    // Build wrapper body
    llvm::BasicBlock *EntryBB = llvm::BasicBlock::Create(Context, "entry", Wrapper);
    llvm::IRBuilder<> Builder(EntryBB);

    // Load the stored function pointer
    llvm::Value *StoredFnPtr = Builder.CreateLoad(PtrTy, FnPtrGlobal);

    // Prepare arguments for the call (only pass what the source function expects)
    llvm::SmallVector<llvm::Value *, 8> CallArgs;
    auto ArgIt = Wrapper->arg_begin();
    for (unsigned i = 0; i < SrcParams && ArgIt != Wrapper->arg_end(); ++i, ++ArgIt) {
      llvm::Value *A = &*ArgIt;
      if (A->getType() != SrcFnType->getParamType(i))
        A = Builder.CreateBitOrPointerCast(A, SrcFnType->getParamType(i));
      CallArgs.push_back(A);
    }

    // Call the source function
    llvm::CallInst *Call = Builder.CreateCall(SrcFnType, StoredFnPtr, CallArgs);

    // Return the result
    if (DstFnType->getReturnType()->isVoidTy()) {
      Builder.CreateRetVoid();
    } else {
      llvm::Value *Ret = Call;
      if (Ret->getType() != DstFnType->getReturnType())
        Ret = Builder.CreateBitOrPointerCast(Ret, DstFnType->getReturnType());
      Builder.CreateRet(Ret);
    }

    RuntimeWrapperCache[Key] = Wrapper;
  }

  // Store the function pointer in the global variable
  CharUnits Alignment = CGF.CGM.getPointerAlign();
  Address GlobalAddr(FnPtrGlobal, PtrTy, Alignment);

  // Store the function pointer to be used by the wrapper
  CGF.Builder.CreateStore(FnPtr, GlobalAddr);

  // Return the wrapper function
  return Wrapper;
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
