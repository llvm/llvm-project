//===----- CGHLSLRuntime.cpp - Interface to HLSL Runtimes -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides an abstract class for HLSL code generation.  Concrete
// subclasses of this implement code generation for specific HLSL
// runtime libraries.
//
//===----------------------------------------------------------------------===//

#include "CGHLSLRuntime.h"
#include "CGDebugInfo.h"
#include "CGRecordLayout.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "HLSLBufferLayoutBuilder.h"
#include "TargetInfo.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attrs.inc"
#include "clang/AST/Decl.h"
#include "clang/AST/HLSLResource.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Type.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Frontend/HLSL/RootSignatureMetadata.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include <cstdint>
#include <optional>

using namespace clang;
using namespace CodeGen;
using namespace clang::hlsl;
using namespace llvm;

using llvm::hlsl::CBufferRowSizeInBytes;

namespace {

void addDxilValVersion(StringRef ValVersionStr, llvm::Module &M) {
  // The validation of ValVersionStr is done at HLSLToolChain::TranslateArgs.
  // Assume ValVersionStr is legal here.
  VersionTuple Version;
  if (Version.tryParse(ValVersionStr) || Version.getBuild() ||
      Version.getSubminor() || !Version.getMinor()) {
    return;
  }

  uint64_t Major = Version.getMajor();
  uint64_t Minor = *Version.getMinor();

  auto &Ctx = M.getContext();
  IRBuilder<> B(M.getContext());
  MDNode *Val = MDNode::get(Ctx, {ConstantAsMetadata::get(B.getInt32(Major)),
                                  ConstantAsMetadata::get(B.getInt32(Minor))});
  StringRef DXILValKey = "dx.valver";
  auto *DXILValMD = M.getOrInsertNamedMetadata(DXILValKey);
  DXILValMD->addOperand(Val);
}

void addRootSignatureMD(llvm::dxbc::RootSignatureVersion RootSigVer,
                        ArrayRef<llvm::hlsl::rootsig::RootElement> Elements,
                        llvm::Function *Fn, llvm::Module &M) {
  auto &Ctx = M.getContext();

  llvm::hlsl::rootsig::MetadataBuilder RSBuilder(Ctx, Elements);
  MDNode *RootSignature = RSBuilder.BuildRootSignature();

  ConstantAsMetadata *Version = ConstantAsMetadata::get(ConstantInt::get(
      llvm::Type::getInt32Ty(Ctx), llvm::to_underlying(RootSigVer)));
  ValueAsMetadata *EntryFunc = Fn ? ValueAsMetadata::get(Fn) : nullptr;
  MDNode *MDVals = MDNode::get(Ctx, {EntryFunc, RootSignature, Version});

  StringRef RootSignatureValKey = "dx.rootsignatures";
  auto *RootSignatureValMD = M.getOrInsertNamedMetadata(RootSignatureValKey);
  RootSignatureValMD->addOperand(MDVals);
}

// Find array variable declaration from nested array subscript AST nodes
static const ValueDecl *getArrayDecl(const ArraySubscriptExpr *ASE) {
  const Expr *E = nullptr;
  while (ASE != nullptr) {
    E = ASE->getBase()->IgnoreImpCasts();
    if (!E)
      return nullptr;
    ASE = dyn_cast<ArraySubscriptExpr>(E);
  }
  if (const DeclRefExpr *DRE = dyn_cast_or_null<DeclRefExpr>(E))
    return DRE->getDecl();
  return nullptr;
}

// Get the total size of the array, or -1 if the array is unbounded.
static int getTotalArraySize(ASTContext &AST, const clang::Type *Ty) {
  Ty = Ty->getUnqualifiedDesugaredType();
  assert(Ty->isArrayType() && "expected array type");
  if (Ty->isIncompleteArrayType())
    return -1;
  return AST.getConstantArrayElementCount(cast<ConstantArrayType>(Ty));
}

static Value *buildNameForResource(llvm::StringRef BaseName,
                                   CodeGenModule &CGM) {
  llvm::SmallString<64> GlobalName = {BaseName, ".str"};
  return CGM.GetAddrOfConstantCString(BaseName.str(), GlobalName.c_str())
      .getPointer();
}

static CXXMethodDecl *lookupMethod(CXXRecordDecl *Record, StringRef Name,
                                   StorageClass SC = SC_None) {
  for (auto *Method : Record->methods()) {
    if (Method->getStorageClass() == SC && Method->getName() == Name)
      return Method;
  }
  return nullptr;
}

static CXXMethodDecl *lookupResourceInitMethodAndSetupArgs(
    CodeGenModule &CGM, CXXRecordDecl *ResourceDecl, llvm::Value *Range,
    llvm::Value *Index, StringRef Name, ResourceBindingAttrs &Binding,
    CallArgList &Args) {
  assert(Binding.hasBinding() && "at least one binding attribute expected");

  ASTContext &AST = CGM.getContext();
  CXXMethodDecl *CreateMethod = nullptr;
  Value *NameStr = buildNameForResource(Name, CGM);
  Value *Space = llvm::ConstantInt::get(CGM.IntTy, Binding.getSpace());

  if (Binding.isExplicit()) {
    // explicit binding
    auto *RegSlot = llvm::ConstantInt::get(CGM.IntTy, Binding.getSlot());
    Args.add(RValue::get(RegSlot), AST.UnsignedIntTy);
    const char *Name = Binding.hasCounterImplicitOrderID()
                           ? "__createFromBindingWithImplicitCounter"
                           : "__createFromBinding";
    CreateMethod = lookupMethod(ResourceDecl, Name, SC_Static);
  } else {
    // implicit binding
    auto *OrderID =
        llvm::ConstantInt::get(CGM.IntTy, Binding.getImplicitOrderID());
    Args.add(RValue::get(OrderID), AST.UnsignedIntTy);
    const char *Name = Binding.hasCounterImplicitOrderID()
                           ? "__createFromImplicitBindingWithImplicitCounter"
                           : "__createFromImplicitBinding";
    CreateMethod = lookupMethod(ResourceDecl, Name, SC_Static);
  }
  Args.add(RValue::get(Space), AST.UnsignedIntTy);
  Args.add(RValue::get(Range), AST.IntTy);
  Args.add(RValue::get(Index), AST.UnsignedIntTy);
  Args.add(RValue::get(NameStr), AST.getPointerType(AST.CharTy.withConst()));
  if (Binding.hasCounterImplicitOrderID()) {
    uint32_t CounterBinding = Binding.getCounterImplicitOrderID();
    auto *CounterOrderID = llvm::ConstantInt::get(CGM.IntTy, CounterBinding);
    Args.add(RValue::get(CounterOrderID), AST.UnsignedIntTy);
  }

  return CreateMethod;
}

static void callResourceInitMethod(CodeGenFunction &CGF,
                                   CXXMethodDecl *CreateMethod,
                                   CallArgList &Args, Address ReturnAddress) {
  llvm::Constant *CalleeFn = CGF.CGM.GetAddrOfFunction(CreateMethod);
  const FunctionProtoType *Proto =
      CreateMethod->getType()->getAs<FunctionProtoType>();
  const CGFunctionInfo &FnInfo =
      CGF.CGM.getTypes().arrangeFreeFunctionCall(Args, Proto, false);
  ReturnValueSlot ReturnValue(ReturnAddress, false);
  CGCallee Callee(CGCalleeInfo(Proto), CalleeFn);
  CGF.EmitCall(FnInfo, Callee, ReturnValue, Args, nullptr);
}

// Initializes local resource array variable. For multi-dimensional arrays it
// calls itself recursively to initialize its sub-arrays. The Index used in the
// resource constructor calls will begin at StartIndex and will be incremented
// for each array element. The last used resource Index is returned to the
// caller. If the function returns std::nullopt, it indicates an error.
static std::optional<llvm::Value *> initializeLocalResourceArray(
    CodeGenFunction &CGF, CXXRecordDecl *ResourceDecl,
    const ConstantArrayType *ArrayTy, AggValueSlot &ValueSlot,
    llvm::Value *Range, llvm::Value *StartIndex, StringRef ResourceName,
    ResourceBindingAttrs &Binding, ArrayRef<llvm::Value *> PrevGEPIndices,
    SourceLocation ArraySubsExprLoc) {

  ASTContext &AST = CGF.getContext();
  llvm::IntegerType *IntTy = CGF.CGM.IntTy;
  llvm::Value *Index = StartIndex;
  llvm::Value *One = llvm::ConstantInt::get(IntTy, 1);
  const uint64_t ArraySize = ArrayTy->getSExtSize();
  QualType ElemType = ArrayTy->getElementType();
  Address TmpArrayAddr = ValueSlot.getAddress();

  // Add additional index to the getelementptr call indices.
  // This index will be updated for each array element in the loops below.
  SmallVector<llvm::Value *> GEPIndices(PrevGEPIndices);
  GEPIndices.push_back(llvm::ConstantInt::get(IntTy, 0));

  // For array of arrays, recursively initialize the sub-arrays.
  if (ElemType->isArrayType()) {
    const ConstantArrayType *SubArrayTy = cast<ConstantArrayType>(ElemType);
    for (uint64_t I = 0; I < ArraySize; I++) {
      if (I > 0) {
        Index = CGF.Builder.CreateAdd(Index, One);
        GEPIndices.back() = llvm::ConstantInt::get(IntTy, I);
      }
      std::optional<llvm::Value *> MaybeIndex = initializeLocalResourceArray(
          CGF, ResourceDecl, SubArrayTy, ValueSlot, Range, Index, ResourceName,
          Binding, GEPIndices, ArraySubsExprLoc);
      if (!MaybeIndex)
        return std::nullopt;
      Index = *MaybeIndex;
    }
    return Index;
  }

  // For array of resources, initialize each resource in the array.
  llvm::Type *Ty = CGF.ConvertTypeForMem(ElemType);
  CharUnits ElemSize = AST.getTypeSizeInChars(ElemType);
  CharUnits Align =
      TmpArrayAddr.getAlignment().alignmentOfArrayElement(ElemSize);

  for (uint64_t I = 0; I < ArraySize; I++) {
    if (I > 0) {
      Index = CGF.Builder.CreateAdd(Index, One);
      GEPIndices.back() = llvm::ConstantInt::get(IntTy, I);
    }
    Address ReturnAddress =
        CGF.Builder.CreateGEP(TmpArrayAddr, GEPIndices, Ty, Align);

    CallArgList Args;
    CXXMethodDecl *CreateMethod = lookupResourceInitMethodAndSetupArgs(
        CGF.CGM, ResourceDecl, Range, Index, ResourceName, Binding, Args);

    if (!CreateMethod)
      // This can happen if someone creates an array of structs that looks like
      // an HLSL resource record array but it does not have the required static
      // create method. No binding will be generated for it.
      return std::nullopt;

    callResourceInitMethod(CGF, CreateMethod, Args, ReturnAddress);
  }
  return Index;
}

} // namespace

llvm::Type *
CGHLSLRuntime::convertHLSLSpecificType(const Type *T,
                                       const CGHLSLOffsetInfo &OffsetInfo) {
  assert(T->isHLSLSpecificType() && "Not an HLSL specific type!");

  // Check if the target has a specific translation for this type first.
  if (llvm::Type *TargetTy =
          CGM.getTargetCodeGenInfo().getHLSLType(CGM, T, OffsetInfo))
    return TargetTy;

  llvm_unreachable("Generic handling of HLSL types is not supported.");
}

llvm::Triple::ArchType CGHLSLRuntime::getArch() {
  return CGM.getTarget().getTriple().getArch();
}

// Emits constant global variables for buffer constants declarations
// and creates metadata linking the constant globals with the buffer global.
void CGHLSLRuntime::emitBufferGlobalsAndMetadata(
    const HLSLBufferDecl *BufDecl, llvm::GlobalVariable *BufGV,
    const CGHLSLOffsetInfo &OffsetInfo) {
  LLVMContext &Ctx = CGM.getLLVMContext();

  // get the layout struct from constant buffer target type
  llvm::Type *BufType = BufGV->getValueType();
  llvm::StructType *LayoutStruct = cast<llvm::StructType>(
      cast<llvm::TargetExtType>(BufType)->getTypeParameter(0));

  SmallVector<std::pair<VarDecl *, uint32_t>> DeclsWithOffset;
  size_t OffsetIdx = 0;
  for (Decl *D : BufDecl->buffer_decls()) {
    if (isa<CXXRecordDecl, EmptyDecl>(D))
      // Nothing to do for this declaration.
      continue;
    if (isa<FunctionDecl>(D)) {
      // A function within an cbuffer is effectively a top-level function.
      CGM.EmitTopLevelDecl(D);
      continue;
    }
    VarDecl *VD = dyn_cast<VarDecl>(D);
    if (!VD)
      continue;

    QualType VDTy = VD->getType();
    if (VDTy.getAddressSpace() != LangAS::hlsl_constant) {
      if (VD->getStorageClass() == SC_Static ||
          VDTy.getAddressSpace() == LangAS::hlsl_groupshared ||
          VDTy->isHLSLResourceRecord() || VDTy->isHLSLResourceRecordArray()) {
        // Emit static and groupshared variables and resource classes inside
        // cbuffer as regular globals
        CGM.EmitGlobal(VD);
      } else {
        // Anything else that is not in the hlsl_constant address space must be
        // an empty struct or a zero-sized array and can be ignored
        assert(BufDecl->getASTContext().getTypeSize(VDTy) == 0 &&
               "constant buffer decl with non-zero sized type outside of "
               "hlsl_constant address space");
      }
      continue;
    }

    DeclsWithOffset.emplace_back(VD, OffsetInfo[OffsetIdx++]);
  }

  if (!OffsetInfo.empty())
    llvm::stable_sort(DeclsWithOffset, [](const auto &LHS, const auto &RHS) {
      return CGHLSLOffsetInfo::compareOffsets(LHS.second, RHS.second);
    });

  // Associate the buffer global variable with its constants
  SmallVector<llvm::Metadata *> BufGlobals;
  BufGlobals.reserve(DeclsWithOffset.size() + 1);
  BufGlobals.push_back(ValueAsMetadata::get(BufGV));

  auto ElemIt = LayoutStruct->element_begin();
  for (auto &[VD, _] : DeclsWithOffset) {
    if (CGM.getTargetCodeGenInfo().isHLSLPadding(*ElemIt))
      ++ElemIt;

    assert(ElemIt != LayoutStruct->element_end() &&
           "number of elements in layout struct does not match");
    llvm::Type *LayoutType = *ElemIt++;

    GlobalVariable *ElemGV =
        cast<GlobalVariable>(CGM.GetAddrOfGlobalVar(VD, LayoutType));
    BufGlobals.push_back(ValueAsMetadata::get(ElemGV));
  }
  assert(ElemIt == LayoutStruct->element_end() &&
         "number of elements in layout struct does not match");

  // add buffer metadata to the module
  CGM.getModule()
      .getOrInsertNamedMetadata("hlsl.cbs")
      ->addOperand(MDNode::get(Ctx, BufGlobals));
}

// Creates resource handle type for the HLSL buffer declaration
static const clang::HLSLAttributedResourceType *
createBufferHandleType(const HLSLBufferDecl *BufDecl) {
  ASTContext &AST = BufDecl->getASTContext();
  QualType QT = AST.getHLSLAttributedResourceType(
      AST.HLSLResourceTy, AST.getCanonicalTagType(BufDecl->getLayoutStruct()),
      HLSLAttributedResourceType::Attributes(ResourceClass::CBuffer));
  return cast<HLSLAttributedResourceType>(QT.getTypePtr());
}

CGHLSLOffsetInfo CGHLSLOffsetInfo::fromDecl(const HLSLBufferDecl &BufDecl) {
  CGHLSLOffsetInfo Result;

  // If we don't have packoffset info, just return an empty result.
  if (!BufDecl.hasValidPackoffset())
    return Result;

  for (Decl *D : BufDecl.buffer_decls()) {
    if (isa<CXXRecordDecl, EmptyDecl>(D) || isa<FunctionDecl>(D)) {
      continue;
    }
    VarDecl *VD = dyn_cast<VarDecl>(D);
    if (!VD || VD->getType().getAddressSpace() != LangAS::hlsl_constant)
      continue;

    if (!VD->hasAttrs()) {
      Result.Offsets.push_back(Unspecified);
      continue;
    }

    uint32_t Offset = Unspecified;
    for (auto *Attr : VD->getAttrs()) {
      if (auto *POA = dyn_cast<HLSLPackOffsetAttr>(Attr)) {
        Offset = POA->getOffsetInBytes();
        break;
      }
      auto *RBA = dyn_cast<HLSLResourceBindingAttr>(Attr);
      if (RBA &&
          RBA->getRegisterType() == HLSLResourceBindingAttr::RegisterType::C) {
        Offset = RBA->getSlotNumber() * CBufferRowSizeInBytes;
        break;
      }
    }
    Result.Offsets.push_back(Offset);
  }
  return Result;
}

// Codegen for HLSLBufferDecl
void CGHLSLRuntime::addBuffer(const HLSLBufferDecl *BufDecl) {

  assert(BufDecl->isCBuffer() && "tbuffer codegen is not supported yet");

  // create resource handle type for the buffer
  const clang::HLSLAttributedResourceType *ResHandleTy =
      createBufferHandleType(BufDecl);

  // empty constant buffer is ignored
  if (ResHandleTy->getContainedType()->getAsCXXRecordDecl()->isEmpty())
    return;

  // create global variable for the constant buffer
  CGHLSLOffsetInfo OffsetInfo = CGHLSLOffsetInfo::fromDecl(*BufDecl);
  llvm::Type *LayoutTy = convertHLSLSpecificType(ResHandleTy, OffsetInfo);
  llvm::GlobalVariable *BufGV = new GlobalVariable(
      LayoutTy, /*isConstant*/ false,
      GlobalValue::LinkageTypes::ExternalLinkage, PoisonValue::get(LayoutTy),
      llvm::formatv("{0}{1}", BufDecl->getName(),
                    BufDecl->isCBuffer() ? ".cb" : ".tb"),
      GlobalValue::NotThreadLocal);
  CGM.getModule().insertGlobalVariable(BufGV);

  // Add globals for constant buffer elements and create metadata nodes
  emitBufferGlobalsAndMetadata(BufDecl, BufGV, OffsetInfo);

  // Initialize cbuffer from binding (implicit or explicit)
  initializeBufferFromBinding(BufDecl, BufGV);
}

void CGHLSLRuntime::addRootSignature(
    const HLSLRootSignatureDecl *SignatureDecl) {
  llvm::Module &M = CGM.getModule();
  Triple T(M.getTargetTriple());

  // Generated later with the function decl if not targeting root signature
  if (T.getEnvironment() != Triple::EnvironmentType::RootSignature)
    return;

  addRootSignatureMD(SignatureDecl->getVersion(),
                     SignatureDecl->getRootElements(), nullptr, M);
}

llvm::StructType *
CGHLSLRuntime::getHLSLBufferLayoutType(const RecordType *StructType) {
  const auto Entry = LayoutTypes.find(StructType);
  if (Entry != LayoutTypes.end())
    return Entry->getSecond();
  return nullptr;
}

void CGHLSLRuntime::addHLSLBufferLayoutType(const RecordType *StructType,
                                            llvm::StructType *LayoutTy) {
  assert(getHLSLBufferLayoutType(StructType) == nullptr &&
         "layout type for this struct already exist");
  LayoutTypes[StructType] = LayoutTy;
}

void CGHLSLRuntime::finishCodeGen() {
  auto &TargetOpts = CGM.getTarget().getTargetOpts();
  auto &CodeGenOpts = CGM.getCodeGenOpts();
  auto &LangOpts = CGM.getLangOpts();
  llvm::Module &M = CGM.getModule();
  Triple T(M.getTargetTriple());
  if (T.getArch() == Triple::ArchType::dxil)
    addDxilValVersion(TargetOpts.DxilValidatorVersion, M);
  if (CodeGenOpts.ResMayAlias)
    M.setModuleFlag(llvm::Module::ModFlagBehavior::Error, "dx.resmayalias", 1);

  // NativeHalfType corresponds to the -fnative-half-type clang option which is
  // aliased by clang-dxc's -enable-16bit-types option. This option is used to
  // set the UseNativeLowPrecision DXIL module flag in the DirectX backend
  if (LangOpts.NativeHalfType)
    M.setModuleFlag(llvm::Module::ModFlagBehavior::Error, "dx.nativelowprec",
                    1);

  generateGlobalCtorDtorCalls();
}

void clang::CodeGen::CGHLSLRuntime::setHLSLEntryAttributes(
    const FunctionDecl *FD, llvm::Function *Fn) {
  const auto *ShaderAttr = FD->getAttr<HLSLShaderAttr>();
  assert(ShaderAttr && "All entry functions must have a HLSLShaderAttr");
  const StringRef ShaderAttrKindStr = "hlsl.shader";
  Fn->addFnAttr(ShaderAttrKindStr,
                llvm::Triple::getEnvironmentTypeName(ShaderAttr->getType()));
  if (HLSLNumThreadsAttr *NumThreadsAttr = FD->getAttr<HLSLNumThreadsAttr>()) {
    const StringRef NumThreadsKindStr = "hlsl.numthreads";
    std::string NumThreadsStr =
        formatv("{0},{1},{2}", NumThreadsAttr->getX(), NumThreadsAttr->getY(),
                NumThreadsAttr->getZ());
    Fn->addFnAttr(NumThreadsKindStr, NumThreadsStr);
  }
  if (HLSLWaveSizeAttr *WaveSizeAttr = FD->getAttr<HLSLWaveSizeAttr>()) {
    const StringRef WaveSizeKindStr = "hlsl.wavesize";
    std::string WaveSizeStr =
        formatv("{0},{1},{2}", WaveSizeAttr->getMin(), WaveSizeAttr->getMax(),
                WaveSizeAttr->getPreferred());
    Fn->addFnAttr(WaveSizeKindStr, WaveSizeStr);
  }
  // HLSL entry functions are materialized for module functions with
  // HLSLShaderAttr attribute. SetLLVMFunctionAttributesForDefinition called
  // later in the compiler-flow for such module functions is not aware of and
  // hence not able to set attributes of the newly materialized entry functions.
  // So, set attributes of entry function here, as appropriate.
  if (CGM.getCodeGenOpts().OptimizationLevel == 0)
    Fn->addFnAttr(llvm::Attribute::OptimizeNone);
  Fn->addFnAttr(llvm::Attribute::NoInline);

  if (CGM.getLangOpts().HLSLSpvEnableMaximalReconvergence) {
    Fn->addFnAttr("enable-maximal-reconvergence", "true");
  }
}

static Value *buildVectorInput(IRBuilder<> &B, Function *F, llvm::Type *Ty) {
  if (const auto *VT = dyn_cast<FixedVectorType>(Ty)) {
    Value *Result = PoisonValue::get(Ty);
    for (unsigned I = 0; I < VT->getNumElements(); ++I) {
      Value *Elt = B.CreateCall(F, {B.getInt32(I)});
      Result = B.CreateInsertElement(Result, Elt, I);
    }
    return Result;
  }
  return B.CreateCall(F, {B.getInt32(0)});
}

static void addSPIRVBuiltinDecoration(llvm::GlobalVariable *GV,
                                      unsigned BuiltIn) {
  LLVMContext &Ctx = GV->getContext();
  IRBuilder<> B(GV->getContext());
  MDNode *Operands = MDNode::get(
      Ctx,
      {ConstantAsMetadata::get(B.getInt32(/* Spirv::Decoration::BuiltIn */ 11)),
       ConstantAsMetadata::get(B.getInt32(BuiltIn))});
  MDNode *Decoration = MDNode::get(Ctx, {Operands});
  GV->addMetadata("spirv.Decorations", *Decoration);
}

static void addLocationDecoration(llvm::GlobalVariable *GV, unsigned Location) {
  LLVMContext &Ctx = GV->getContext();
  IRBuilder<> B(GV->getContext());
  MDNode *Operands =
      MDNode::get(Ctx, {ConstantAsMetadata::get(B.getInt32(/* Location */ 30)),
                        ConstantAsMetadata::get(B.getInt32(Location))});
  MDNode *Decoration = MDNode::get(Ctx, {Operands});
  GV->addMetadata("spirv.Decorations", *Decoration);
}

static llvm::Value *createSPIRVBuiltinLoad(IRBuilder<> &B, llvm::Module &M,
                                           llvm::Type *Ty, const Twine &Name,
                                           unsigned BuiltInID) {
  auto *GV = new llvm::GlobalVariable(
      M, Ty, /* isConstant= */ true, llvm::GlobalValue::ExternalLinkage,
      /* Initializer= */ nullptr, Name, /* insertBefore= */ nullptr,
      llvm::GlobalVariable::GeneralDynamicTLSModel,
      /* AddressSpace */ 7, /* isExternallyInitialized= */ true);
  addSPIRVBuiltinDecoration(GV, BuiltInID);
  GV->setVisibility(llvm::GlobalValue::HiddenVisibility);
  return B.CreateLoad(Ty, GV);
}

static llvm::Value *createSPIRVLocationLoad(IRBuilder<> &B, llvm::Module &M,
                                            llvm::Type *Ty, unsigned Location,
                                            StringRef Name) {
  auto *GV = new llvm::GlobalVariable(
      M, Ty, /* isConstant= */ true, llvm::GlobalValue::ExternalLinkage,
      /* Initializer= */ nullptr, /* Name= */ Name, /* insertBefore= */ nullptr,
      llvm::GlobalVariable::GeneralDynamicTLSModel,
      /* AddressSpace */ 7, /* isExternallyInitialized= */ true);
  GV->setVisibility(llvm::GlobalValue::HiddenVisibility);
  addLocationDecoration(GV, Location);
  return B.CreateLoad(Ty, GV);
}

llvm::Value *
CGHLSLRuntime::emitSPIRVUserSemanticLoad(llvm::IRBuilder<> &B, llvm::Type *Type,
                                         HLSLAppliedSemanticAttr *Semantic,
                                         std::optional<unsigned> Index) {
  Twine BaseName = Twine(Semantic->getAttrName()->getName());
  Twine VariableName = BaseName.concat(Twine(Index.value_or(0)));

  unsigned Location = SPIRVLastAssignedInputSemanticLocation;

  // DXC completely ignores the semantic/index pair. Location are assigned from
  // the first semantic to the last.
  llvm::ArrayType *AT = dyn_cast<llvm::ArrayType>(Type);
  unsigned ElementCount = AT ? AT->getNumElements() : 1;
  SPIRVLastAssignedInputSemanticLocation += ElementCount;
  return createSPIRVLocationLoad(B, CGM.getModule(), Type, Location,
                                 VariableName.str());
}

llvm::Value *
CGHLSLRuntime::emitDXILUserSemanticLoad(llvm::IRBuilder<> &B, llvm::Type *Type,
                                        HLSLAppliedSemanticAttr *Semantic,
                                        std::optional<unsigned> Index) {
  Twine BaseName = Twine(Semantic->getAttrName()->getName());
  Twine VariableName = BaseName.concat(Twine(Index.value_or(0)));

  // DXIL packing rules etc shall be handled here.
  // FIXME: generate proper sigpoint, index, col, row values.
  // FIXME: also DXIL loads vectors element by element.
  SmallVector<Value *> Args{B.getInt32(4), B.getInt32(0), B.getInt32(0),
                            B.getInt8(0),
                            llvm::PoisonValue::get(B.getInt32Ty())};

  llvm::Intrinsic::ID IntrinsicID = llvm::Intrinsic::dx_load_input;
  llvm::Value *Value = B.CreateIntrinsic(/*ReturnType=*/Type, IntrinsicID, Args,
                                         nullptr, VariableName);
  return Value;
}

llvm::Value *CGHLSLRuntime::emitUserSemanticLoad(
    IRBuilder<> &B, llvm::Type *Type, const clang::DeclaratorDecl *Decl,
    HLSLAppliedSemanticAttr *Semantic, std::optional<unsigned> Index) {
  if (CGM.getTarget().getTriple().isSPIRV())
    return emitSPIRVUserSemanticLoad(B, Type, Semantic, Index);

  if (CGM.getTarget().getTriple().isDXIL())
    return emitDXILUserSemanticLoad(B, Type, Semantic, Index);

  llvm_unreachable("Unsupported target for user-semantic load.");
}

llvm::Value *CGHLSLRuntime::emitSystemSemanticLoad(
    IRBuilder<> &B, llvm::Type *Type, const clang::DeclaratorDecl *Decl,
    HLSLAppliedSemanticAttr *Semantic, std::optional<unsigned> Index) {

  std::string SemanticName = Semantic->getAttrName()->getName().upper();
  if (SemanticName == "SV_GROUPINDEX") {
    llvm::Function *GroupIndex =
        CGM.getIntrinsic(getFlattenedThreadIdInGroupIntrinsic());
    return B.CreateCall(FunctionCallee(GroupIndex));
  }

  if (SemanticName == "SV_DISPATCHTHREADID") {
    llvm::Intrinsic::ID IntrinID = getThreadIdIntrinsic();
    llvm::Function *ThreadIDIntrinsic =
        llvm::Intrinsic::isOverloaded(IntrinID)
            ? CGM.getIntrinsic(IntrinID, {CGM.Int32Ty})
            : CGM.getIntrinsic(IntrinID);
    return buildVectorInput(B, ThreadIDIntrinsic, Type);
  }

  if (SemanticName == "SV_GROUPTHREADID") {
    llvm::Intrinsic::ID IntrinID = getGroupThreadIdIntrinsic();
    llvm::Function *GroupThreadIDIntrinsic =
        llvm::Intrinsic::isOverloaded(IntrinID)
            ? CGM.getIntrinsic(IntrinID, {CGM.Int32Ty})
            : CGM.getIntrinsic(IntrinID);
    return buildVectorInput(B, GroupThreadIDIntrinsic, Type);
  }

  if (SemanticName == "SV_GROUPID") {
    llvm::Intrinsic::ID IntrinID = getGroupIdIntrinsic();
    llvm::Function *GroupIDIntrinsic =
        llvm::Intrinsic::isOverloaded(IntrinID)
            ? CGM.getIntrinsic(IntrinID, {CGM.Int32Ty})
            : CGM.getIntrinsic(IntrinID);
    return buildVectorInput(B, GroupIDIntrinsic, Type);
  }

  if (SemanticName == "SV_POSITION") {
    if (CGM.getTriple().getEnvironment() == Triple::EnvironmentType::Pixel)
      return createSPIRVBuiltinLoad(B, CGM.getModule(), Type,
                                    Semantic->getAttrName()->getName(),
                                    /* BuiltIn::FragCoord */ 15);
  }

  llvm_unreachable("non-handled system semantic. FIXME.");
}

llvm::Value *CGHLSLRuntime::handleScalarSemanticLoad(
    IRBuilder<> &B, const FunctionDecl *FD, llvm::Type *Type,
    const clang::DeclaratorDecl *Decl, HLSLAppliedSemanticAttr *Semantic) {

  std::optional<unsigned> Index = Semantic->getSemanticIndex();
  if (Semantic->getAttrName()->getName().starts_with_insensitive("SV_"))
    return emitSystemSemanticLoad(B, Type, Decl, Semantic, Index);
  return emitUserSemanticLoad(B, Type, Decl, Semantic, Index);
}

std::pair<llvm::Value *, specific_attr_iterator<HLSLAppliedSemanticAttr>>
CGHLSLRuntime::handleStructSemanticLoad(
    IRBuilder<> &B, const FunctionDecl *FD, llvm::Type *Type,
    const clang::DeclaratorDecl *Decl,
    specific_attr_iterator<HLSLAppliedSemanticAttr> AttrBegin,
    specific_attr_iterator<HLSLAppliedSemanticAttr> AttrEnd) {
  const llvm::StructType *ST = cast<StructType>(Type);
  const clang::RecordDecl *RD = Decl->getType()->getAsRecordDecl();

  assert(std::distance(RD->field_begin(), RD->field_end()) ==
         ST->getNumElements());

  llvm::Value *Aggregate = llvm::PoisonValue::get(Type);
  auto FieldDecl = RD->field_begin();
  for (unsigned I = 0; I < ST->getNumElements(); ++I) {
    auto [ChildValue, NextAttr] = handleSemanticLoad(
        B, FD, ST->getElementType(I), *FieldDecl, AttrBegin, AttrEnd);
    AttrBegin = NextAttr;
    assert(ChildValue);
    Aggregate = B.CreateInsertValue(Aggregate, ChildValue, I);
    ++FieldDecl;
  }

  return std::make_pair(Aggregate, AttrBegin);
}

std::pair<llvm::Value *, specific_attr_iterator<HLSLAppliedSemanticAttr>>
CGHLSLRuntime::handleSemanticLoad(
    IRBuilder<> &B, const FunctionDecl *FD, llvm::Type *Type,
    const clang::DeclaratorDecl *Decl,
    specific_attr_iterator<HLSLAppliedSemanticAttr> AttrBegin,
    specific_attr_iterator<HLSLAppliedSemanticAttr> AttrEnd) {
  assert(AttrBegin != AttrEnd);
  if (Type->isStructTy())
    return handleStructSemanticLoad(B, FD, Type, Decl, AttrBegin, AttrEnd);

  HLSLAppliedSemanticAttr *Attr = *AttrBegin;
  ++AttrBegin;
  return std::make_pair(handleScalarSemanticLoad(B, FD, Type, Decl, Attr),
                        AttrBegin);
}

void CGHLSLRuntime::emitEntryFunction(const FunctionDecl *FD,
                                      llvm::Function *Fn) {
  llvm::Module &M = CGM.getModule();
  llvm::LLVMContext &Ctx = M.getContext();
  auto *EntryTy = llvm::FunctionType::get(llvm::Type::getVoidTy(Ctx), false);
  Function *EntryFn =
      Function::Create(EntryTy, Function::ExternalLinkage, FD->getName(), &M);

  // Copy function attributes over, we have no argument or return attributes
  // that can be valid on the real entry.
  AttributeList NewAttrs = AttributeList::get(Ctx, AttributeList::FunctionIndex,
                                              Fn->getAttributes().getFnAttrs());
  EntryFn->setAttributes(NewAttrs);
  setHLSLEntryAttributes(FD, EntryFn);

  // Set the called function as internal linkage.
  Fn->setLinkage(GlobalValue::InternalLinkage);

  BasicBlock *BB = BasicBlock::Create(Ctx, "entry", EntryFn);
  IRBuilder<> B(BB);
  llvm::SmallVector<Value *> Args;

  SmallVector<OperandBundleDef, 1> OB;
  if (CGM.shouldEmitConvergenceTokens()) {
    assert(EntryFn->isConvergent());
    llvm::Value *I =
        B.CreateIntrinsic(llvm::Intrinsic::experimental_convergence_entry, {});
    llvm::Value *bundleArgs[] = {I};
    OB.emplace_back("convergencectrl", bundleArgs);
  }

  // FIXME: support struct parameters where semantics are on members.
  // See: https://github.com/llvm/llvm-project/issues/57874
  unsigned SRetOffset = 0;
  for (const auto &Param : Fn->args()) {
    if (Param.hasStructRetAttr()) {
      // FIXME: support output.
      // See: https://github.com/llvm/llvm-project/issues/57874
      SRetOffset = 1;
      Args.emplace_back(PoisonValue::get(Param.getType()));
      continue;
    }

    const ParmVarDecl *PD = FD->getParamDecl(Param.getArgNo() - SRetOffset);
    llvm::Value *SemanticValue = nullptr;
    if ([[maybe_unused]] HLSLParamModifierAttr *MA =
            PD->getAttr<HLSLParamModifierAttr>()) {
      llvm_unreachable("Not handled yet");
    } else {
      llvm::Type *ParamType =
          Param.hasByValAttr() ? Param.getParamByValType() : Param.getType();
      auto AttrBegin = PD->specific_attr_begin<HLSLAppliedSemanticAttr>();
      auto AttrEnd = PD->specific_attr_end<HLSLAppliedSemanticAttr>();
      auto Result =
          handleSemanticLoad(B, FD, ParamType, PD, AttrBegin, AttrEnd);
      SemanticValue = Result.first;
      if (!SemanticValue)
        return;
      if (Param.hasByValAttr()) {
        llvm::Value *Var = B.CreateAlloca(Param.getParamByValType());
        B.CreateStore(SemanticValue, Var);
        SemanticValue = Var;
      }
    }

    assert(SemanticValue);
    Args.push_back(SemanticValue);
  }

  CallInst *CI = B.CreateCall(FunctionCallee(Fn), Args, OB);
  CI->setCallingConv(Fn->getCallingConv());
  // FIXME: Handle codegen for return type semantics.
  // See: https://github.com/llvm/llvm-project/issues/57875
  B.CreateRetVoid();

  // Add and identify root signature to function, if applicable
  for (const Attr *Attr : FD->getAttrs()) {
    if (const auto *RSAttr = dyn_cast<RootSignatureAttr>(Attr)) {
      auto *RSDecl = RSAttr->getSignatureDecl();
      addRootSignatureMD(RSDecl->getVersion(), RSDecl->getRootElements(),
                         EntryFn, M);
    }
  }
}

static void gatherFunctions(SmallVectorImpl<Function *> &Fns, llvm::Module &M,
                            bool CtorOrDtor) {
  const auto *GV =
      M.getNamedGlobal(CtorOrDtor ? "llvm.global_ctors" : "llvm.global_dtors");
  if (!GV)
    return;
  const auto *CA = dyn_cast<ConstantArray>(GV->getInitializer());
  if (!CA)
    return;
  // The global_ctor array elements are a struct [Priority, Fn *, COMDat].
  // HLSL neither supports priorities or COMDat values, so we will check those
  // in an assert but not handle them.

  for (const auto &Ctor : CA->operands()) {
    if (isa<ConstantAggregateZero>(Ctor))
      continue;
    ConstantStruct *CS = cast<ConstantStruct>(Ctor);

    assert(cast<ConstantInt>(CS->getOperand(0))->getValue() == 65535 &&
           "HLSL doesn't support setting priority for global ctors.");
    assert(isa<ConstantPointerNull>(CS->getOperand(2)) &&
           "HLSL doesn't support COMDat for global ctors.");
    Fns.push_back(cast<Function>(CS->getOperand(1)));
  }
}

void CGHLSLRuntime::generateGlobalCtorDtorCalls() {
  llvm::Module &M = CGM.getModule();
  SmallVector<Function *> CtorFns;
  SmallVector<Function *> DtorFns;
  gatherFunctions(CtorFns, M, true);
  gatherFunctions(DtorFns, M, false);

  // Insert a call to the global constructor at the beginning of the entry block
  // to externally exported functions. This is a bit of a hack, but HLSL allows
  // global constructors, but doesn't support driver initialization of globals.
  for (auto &F : M.functions()) {
    if (!F.hasFnAttribute("hlsl.shader"))
      continue;
    auto *Token = getConvergenceToken(F.getEntryBlock());
    Instruction *IP = &*F.getEntryBlock().begin();
    SmallVector<OperandBundleDef, 1> OB;
    if (Token) {
      llvm::Value *bundleArgs[] = {Token};
      OB.emplace_back("convergencectrl", bundleArgs);
      IP = Token->getNextNode();
    }
    IRBuilder<> B(IP);
    for (auto *Fn : CtorFns) {
      auto CI = B.CreateCall(FunctionCallee(Fn), {}, OB);
      CI->setCallingConv(Fn->getCallingConv());
    }

    // Insert global dtors before the terminator of the last instruction
    B.SetInsertPoint(F.back().getTerminator());
    for (auto *Fn : DtorFns) {
      auto CI = B.CreateCall(FunctionCallee(Fn), {}, OB);
      CI->setCallingConv(Fn->getCallingConv());
    }
  }

  // No need to keep global ctors/dtors for non-lib profile after call to
  // ctors/dtors added for entry.
  Triple T(M.getTargetTriple());
  if (T.getEnvironment() != Triple::EnvironmentType::Library) {
    if (auto *GV = M.getNamedGlobal("llvm.global_ctors"))
      GV->eraseFromParent();
    if (auto *GV = M.getNamedGlobal("llvm.global_dtors"))
      GV->eraseFromParent();
  }
}

static void initializeBuffer(CodeGenModule &CGM, llvm::GlobalVariable *GV,
                             Intrinsic::ID IntrID,
                             ArrayRef<llvm::Value *> Args) {

  LLVMContext &Ctx = CGM.getLLVMContext();
  llvm::Function *InitResFunc = llvm::Function::Create(
      llvm::FunctionType::get(CGM.VoidTy, false),
      llvm::GlobalValue::InternalLinkage,
      ("_init_buffer_" + GV->getName()).str(), CGM.getModule());
  InitResFunc->addFnAttr(llvm::Attribute::AlwaysInline);

  llvm::BasicBlock *EntryBB =
      llvm::BasicBlock::Create(Ctx, "entry", InitResFunc);
  CGBuilderTy Builder(CGM, Ctx);
  const DataLayout &DL = CGM.getModule().getDataLayout();
  Builder.SetInsertPoint(EntryBB);

  // Make sure the global variable is buffer resource handle
  llvm::Type *HandleTy = GV->getValueType();
  assert(HandleTy->isTargetExtTy() && "unexpected type of the buffer global");

  llvm::Value *CreateHandle = Builder.CreateIntrinsic(
      /*ReturnType=*/HandleTy, IntrID, Args, nullptr,
      Twine(GV->getName()).concat("_h"));

  llvm::Value *HandleRef = Builder.CreateStructGEP(GV->getValueType(), GV, 0);
  Builder.CreateAlignedStore(CreateHandle, HandleRef,
                             HandleRef->getPointerAlignment(DL));
  Builder.CreateRetVoid();

  CGM.AddCXXGlobalInit(InitResFunc);
}

void CGHLSLRuntime::initializeBufferFromBinding(const HLSLBufferDecl *BufDecl,
                                                llvm::GlobalVariable *GV) {
  ResourceBindingAttrs Binding(BufDecl);
  assert(Binding.hasBinding() &&
         "cbuffer/tbuffer should always have resource binding attribute");

  auto *Index = llvm::ConstantInt::get(CGM.IntTy, 0);
  auto *RangeSize = llvm::ConstantInt::get(CGM.IntTy, 1);
  auto *Space = llvm::ConstantInt::get(CGM.IntTy, Binding.getSpace());
  Value *Name = buildNameForResource(BufDecl->getName(), CGM);

  // buffer with explicit binding
  if (Binding.isExplicit()) {
    llvm::Intrinsic::ID IntrinsicID =
        CGM.getHLSLRuntime().getCreateHandleFromBindingIntrinsic();
    auto *RegSlot = llvm::ConstantInt::get(CGM.IntTy, Binding.getSlot());
    SmallVector<Value *> Args{Space, RegSlot, RangeSize, Index, Name};
    initializeBuffer(CGM, GV, IntrinsicID, Args);
  } else {
    // buffer with implicit binding
    llvm::Intrinsic::ID IntrinsicID =
        CGM.getHLSLRuntime().getCreateHandleFromImplicitBindingIntrinsic();
    auto *OrderID =
        llvm::ConstantInt::get(CGM.IntTy, Binding.getImplicitOrderID());
    SmallVector<Value *> Args{OrderID, Space, RangeSize, Index, Name};
    initializeBuffer(CGM, GV, IntrinsicID, Args);
  }
}

void CGHLSLRuntime::handleGlobalVarDefinition(const VarDecl *VD,
                                              llvm::GlobalVariable *GV) {
  if (auto Attr = VD->getAttr<HLSLVkExtBuiltinInputAttr>())
    addSPIRVBuiltinDecoration(GV, Attr->getBuiltIn());
}

llvm::Instruction *CGHLSLRuntime::getConvergenceToken(BasicBlock &BB) {
  if (!CGM.shouldEmitConvergenceTokens())
    return nullptr;

  auto E = BB.end();
  for (auto I = BB.begin(); I != E; ++I) {
    auto *II = dyn_cast<llvm::IntrinsicInst>(&*I);
    if (II && llvm::isConvergenceControlIntrinsic(II->getIntrinsicID())) {
      return II;
    }
  }
  llvm_unreachable("Convergence token should have been emitted.");
  return nullptr;
}

class OpaqueValueVisitor : public RecursiveASTVisitor<OpaqueValueVisitor> {
public:
  llvm::SmallVector<OpaqueValueExpr *, 8> OVEs;
  llvm::SmallPtrSet<OpaqueValueExpr *, 8> Visited;
  OpaqueValueVisitor() {}

  bool VisitHLSLOutArgExpr(HLSLOutArgExpr *) {
    // These need to be bound in CodeGenFunction::EmitHLSLOutArgLValues
    // or CodeGenFunction::EmitHLSLOutArgExpr. If they are part of this
    // traversal, the temporary containing the copy out will not have
    // been created yet.
    return false;
  }

  bool VisitOpaqueValueExpr(OpaqueValueExpr *E) {
    // Traverse the source expression first.
    if (E->getSourceExpr())
      TraverseStmt(E->getSourceExpr());

    // Then add this OVE if we haven't seen it before.
    if (Visited.insert(E).second)
      OVEs.push_back(E);

    return true;
  }
};

void CGHLSLRuntime::emitInitListOpaqueValues(CodeGenFunction &CGF,
                                             InitListExpr *E) {

  typedef CodeGenFunction::OpaqueValueMappingData OpaqueValueMappingData;
  OpaqueValueVisitor Visitor;
  Visitor.TraverseStmt(E);
  for (auto *OVE : Visitor.OVEs) {
    if (CGF.isOpaqueValueEmitted(OVE))
      continue;
    if (OpaqueValueMappingData::shouldBindAsLValue(OVE)) {
      LValue LV = CGF.EmitLValue(OVE->getSourceExpr());
      OpaqueValueMappingData::bind(CGF, OVE, LV);
    } else {
      RValue RV = CGF.EmitAnyExpr(OVE->getSourceExpr());
      OpaqueValueMappingData::bind(CGF, OVE, RV);
    }
  }
}

std::optional<LValue> CGHLSLRuntime::emitResourceArraySubscriptExpr(
    const ArraySubscriptExpr *ArraySubsExpr, CodeGenFunction &CGF) {
  assert((ArraySubsExpr->getType()->isHLSLResourceRecord() ||
          ArraySubsExpr->getType()->isHLSLResourceRecordArray()) &&
         "expected resource array subscript expression");

  // Let clang codegen handle local resource array subscripts,
  // or when the subscript references on opaque expression (as part of
  // ArrayInitLoopExpr AST node).
  const VarDecl *ArrayDecl =
      dyn_cast_or_null<VarDecl>(getArrayDecl(ArraySubsExpr));
  if (!ArrayDecl || !ArrayDecl->hasGlobalStorage())
    return std::nullopt;

  // get the resource array type
  ASTContext &AST = ArrayDecl->getASTContext();
  const Type *ResArrayTy = ArrayDecl->getType().getTypePtr();
  assert(ResArrayTy->isHLSLResourceRecordArray() &&
         "expected array of resource classes");

  // Iterate through all nested array subscript expressions to calculate
  // the index in the flattened resource array (if this is a multi-
  // dimensional array). The index is calculated as a sum of all indices
  // multiplied by the total size of the array at that level.
  Value *Index = nullptr;
  const ArraySubscriptExpr *ASE = ArraySubsExpr;
  while (ASE != nullptr) {
    Value *SubIndex = CGF.EmitScalarExpr(ASE->getIdx());
    if (const auto *ArrayTy =
            dyn_cast<ConstantArrayType>(ASE->getType().getTypePtr())) {
      Value *Multiplier = llvm::ConstantInt::get(
          CGM.IntTy, AST.getConstantArrayElementCount(ArrayTy));
      SubIndex = CGF.Builder.CreateMul(SubIndex, Multiplier);
    }
    Index = Index ? CGF.Builder.CreateAdd(Index, SubIndex) : SubIndex;
    ASE = dyn_cast<ArraySubscriptExpr>(ASE->getBase()->IgnoreParenImpCasts());
  }

  // Find binding info for the resource array. For implicit binding
  // an HLSLResourceBindingAttr should have been added by SemaHLSL.
  ResourceBindingAttrs Binding(ArrayDecl);
  assert((Binding.hasBinding()) &&
         "resource array must have a binding attribute");

  // Find the individual resource type.
  QualType ResultTy = ArraySubsExpr->getType();
  QualType ResourceTy =
      ResultTy->isArrayType() ? AST.getBaseElementType(ResultTy) : ResultTy;

  // Create a temporary variable for the result, which is either going
  // to be a single resource instance or a local array of resources (we need to
  // return an LValue).
  RawAddress TmpVar = CGF.CreateMemTemp(ResultTy);
  if (CGF.EmitLifetimeStart(TmpVar.getPointer()))
    CGF.pushFullExprCleanup<CodeGenFunction::CallLifetimeEnd>(
        NormalEHLifetimeMarker, TmpVar);

  AggValueSlot ValueSlot = AggValueSlot::forAddr(
      TmpVar, Qualifiers(), AggValueSlot::IsDestructed_t(true),
      AggValueSlot::DoesNotNeedGCBarriers, AggValueSlot::IsAliased_t(false),
      AggValueSlot::DoesNotOverlap);

  // Calculate total array size (= range size).
  llvm::Value *Range =
      llvm::ConstantInt::get(CGM.IntTy, getTotalArraySize(AST, ResArrayTy));

  // If the result of the subscript operation is a single resource, call the
  // constructor.
  if (ResultTy == ResourceTy) {
    CallArgList Args;
    CXXMethodDecl *CreateMethod = lookupResourceInitMethodAndSetupArgs(
        CGF.CGM, ResourceTy->getAsCXXRecordDecl(), Range, Index,
        ArrayDecl->getName(), Binding, Args);

    if (!CreateMethod)
      // This can happen if someone creates an array of structs that looks like
      // an HLSL resource record array but it does not have the required static
      // create method. No binding will be generated for it.
      return std::nullopt;

    callResourceInitMethod(CGF, CreateMethod, Args, ValueSlot.getAddress());

  } else {
    // The result of the subscript operation is a local resource array which
    // needs to be initialized.
    const ConstantArrayType *ArrayTy =
        cast<ConstantArrayType>(ResultTy.getTypePtr());
    std::optional<llvm::Value *> EndIndex = initializeLocalResourceArray(
        CGF, ResourceTy->getAsCXXRecordDecl(), ArrayTy, ValueSlot, Range, Index,
        ArrayDecl->getName(), Binding, {llvm::ConstantInt::get(CGM.IntTy, 0)},
        ArraySubsExpr->getExprLoc());
    if (!EndIndex)
      return std::nullopt;
  }
  return CGF.MakeAddrLValue(TmpVar, ResultTy, AlignmentSource::Decl);
}

std::optional<LValue> CGHLSLRuntime::emitBufferArraySubscriptExpr(
    const ArraySubscriptExpr *E, CodeGenFunction &CGF,
    llvm::function_ref<llvm::Value *(bool Promote)> EmitIdxAfterBase) {
  // Find the element type to index by first padding the element type per HLSL
  // buffer rules, and then padding out to a 16-byte register boundary if
  // necessary.
  llvm::Type *LayoutTy =
      HLSLBufferLayoutBuilder(CGF.CGM).layOutType(E->getType());
  uint64_t LayoutSizeInBits =
      CGM.getDataLayout().getTypeSizeInBits(LayoutTy).getFixedValue();
  CharUnits ElementSize = CharUnits::fromQuantity(LayoutSizeInBits / 8);
  CharUnits RowAlignedSize = ElementSize.alignTo(CharUnits::fromQuantity(16));
  if (RowAlignedSize > ElementSize) {
    llvm::Type *Padding = CGM.getTargetCodeGenInfo().getHLSLPadding(
        CGM, RowAlignedSize - ElementSize);
    assert(Padding && "No padding type for target?");
    LayoutTy = llvm::StructType::get(CGF.getLLVMContext(), {LayoutTy, Padding},
                                     /*isPacked=*/true);
  }

  // If the layout type doesn't introduce any padding, we don't need to do
  // anything special.
  llvm::Type *OrigTy = CGF.CGM.getTypes().ConvertTypeForMem(E->getType());
  if (LayoutTy == OrigTy)
    return std::nullopt;

  LValueBaseInfo EltBaseInfo;
  TBAAAccessInfo EltTBAAInfo;
  Address Addr =
      CGF.EmitPointerWithAlignment(E->getBase(), &EltBaseInfo, &EltTBAAInfo);
  llvm::Value *Idx = EmitIdxAfterBase(/*Promote*/ true);

  // Index into the object as-if we have an array of the padded element type,
  // and then dereference the element itself to avoid reading padding that may
  // be past the end of the in-memory object.
  SmallVector<llvm::Value *, 2> Indices;
  Indices.push_back(Idx);
  Indices.push_back(llvm::ConstantInt::get(CGF.Int32Ty, 0));

  llvm::Value *GEP = CGF.Builder.CreateGEP(LayoutTy, Addr.emitRawPointer(CGF),
                                           Indices, "cbufferidx");
  Addr = Address(GEP, Addr.getElementType(), RowAlignedSize, KnownNonNull);

  return CGF.MakeAddrLValue(Addr, E->getType(), EltBaseInfo, EltTBAAInfo);
}

namespace {
/// Utility for emitting copies following the HLSL buffer layout rules (ie,
/// copying out of a cbuffer).
class HLSLBufferCopyEmitter {
  CodeGenFunction &CGF;
  Address DestPtr;
  Address SrcPtr;
  llvm::Type *LayoutTy = nullptr;

  SmallVector<llvm::Value *> CurStoreIndices;
  SmallVector<llvm::Value *> CurLoadIndices;

  void emitCopyAtIndices(llvm::Type *FieldTy, llvm::ConstantInt *StoreIndex,
                         llvm::ConstantInt *LoadIndex) {
    CurStoreIndices.push_back(StoreIndex);
    CurLoadIndices.push_back(LoadIndex);
    auto RestoreIndices = llvm::make_scope_exit([&]() {
      CurStoreIndices.pop_back();
      CurLoadIndices.pop_back();
    });

    // First, see if this is some kind of aggregate and recurse.
    if (processArray(FieldTy))
      return;
    if (processBufferLayoutArray(FieldTy))
      return;
    if (processStruct(FieldTy))
      return;

    // When we have a scalar or vector element we can emit the copy.
    CharUnits Align = CharUnits::fromQuantity(
        CGF.CGM.getDataLayout().getABITypeAlign(FieldTy));
    Address SrcGEP = RawAddress(
        CGF.Builder.CreateInBoundsGEP(LayoutTy, SrcPtr.getBasePointer(),
                                      CurLoadIndices, "cbuf.src"),
        FieldTy, Align, SrcPtr.isKnownNonNull());
    Address DestGEP = CGF.Builder.CreateInBoundsGEP(
        DestPtr, CurStoreIndices, FieldTy, Align, "cbuf.dest");
    llvm::Value *Load = CGF.Builder.CreateLoad(SrcGEP, "cbuf.load");
    CGF.Builder.CreateStore(Load, DestGEP);
  }

  bool processArray(llvm::Type *FieldTy) {
    auto *AT = dyn_cast<llvm::ArrayType>(FieldTy);
    if (!AT)
      return false;

    // If we have an llvm::ArrayType this is just a regular array with no top
    // level padding, so all we need to do is copy each member.
    for (unsigned I = 0, E = AT->getNumElements(); I < E; ++I)
      emitCopyAtIndices(AT->getElementType(),
                        llvm::ConstantInt::get(CGF.SizeTy, I),
                        llvm::ConstantInt::get(CGF.SizeTy, I));
    return true;
  }

  bool processBufferLayoutArray(llvm::Type *FieldTy) {
    // A buffer layout array is a struct with two elements: the padded array,
    // and the last element. That is, is should look something like this:
    //
    //   { [%n x { %type, %padding }], %type }
    //
    auto *ST = dyn_cast<llvm::StructType>(FieldTy);
    if (!ST || ST->getNumElements() != 2)
      return false;

    auto *PaddedEltsTy = dyn_cast<llvm::ArrayType>(ST->getElementType(0));
    if (!PaddedEltsTy)
      return false;

    auto *PaddedTy = dyn_cast<llvm::StructType>(PaddedEltsTy->getElementType());
    if (!PaddedTy || PaddedTy->getNumElements() != 2)
      return false;

    if (!CGF.CGM.getTargetCodeGenInfo().isHLSLPadding(
            PaddedTy->getElementType(1)))
      return false;

    llvm::Type *ElementTy = ST->getElementType(1);
    if (PaddedTy->getElementType(0) != ElementTy)
      return false;

    // All but the last of the logical array elements are in the padded array.
    unsigned NumElts = PaddedEltsTy->getNumElements() + 1;

    // Add an extra indirection to the load for the struct and walk the
    // array prefix.
    CurLoadIndices.push_back(llvm::ConstantInt::get(CGF.Int32Ty, 0));
    for (unsigned I = 0; I < NumElts - 1; ++I) {
      // We need to copy the element itself, without the padding.
      CurLoadIndices.push_back(llvm::ConstantInt::get(CGF.SizeTy, I));
      emitCopyAtIndices(ElementTy, llvm::ConstantInt::get(CGF.SizeTy, I),
                        llvm::ConstantInt::get(CGF.Int32Ty, 0));
      CurLoadIndices.pop_back();
    }
    CurLoadIndices.pop_back();

    // Now copy the last element.
    emitCopyAtIndices(ElementTy,
                      llvm::ConstantInt::get(CGF.SizeTy, NumElts - 1),
                      llvm::ConstantInt::get(CGF.Int32Ty, 1));

    return true;
  }

  bool processStruct(llvm::Type *FieldTy) {
    auto *ST = dyn_cast<llvm::StructType>(FieldTy);
    if (!ST)
      return false;

    // Copy the struct field by field, but skip any explicit padding.
    unsigned Skipped = 0;
    for (unsigned I = 0, E = ST->getNumElements(); I < E; ++I) {
      llvm::Type *ElementTy = ST->getElementType(I);
      if (CGF.CGM.getTargetCodeGenInfo().isHLSLPadding(ElementTy))
        ++Skipped;
      else
        emitCopyAtIndices(ElementTy, llvm::ConstantInt::get(CGF.Int32Ty, I),
                          llvm::ConstantInt::get(CGF.Int32Ty, I + Skipped));
    }
    return true;
  }

public:
  HLSLBufferCopyEmitter(CodeGenFunction &CGF, Address DestPtr, Address SrcPtr)
      : CGF(CGF), DestPtr(DestPtr), SrcPtr(SrcPtr) {}

  bool emitCopy(QualType CType) {
    LayoutTy = HLSLBufferLayoutBuilder(CGF.CGM).layOutType(CType);

    // TODO: We should be able to fall back to a regular memcpy if the layout
    // type doesn't have any padding, but that runs into issues in the backend
    // currently.
    //
    // See https://github.com/llvm/wg-hlsl/issues/351
    emitCopyAtIndices(LayoutTy, llvm::ConstantInt::get(CGF.SizeTy, 0),
                      llvm::ConstantInt::get(CGF.SizeTy, 0));
    return true;
  }
};
} // namespace

bool CGHLSLRuntime::emitBufferCopy(CodeGenFunction &CGF, Address DestPtr,
                                   Address SrcPtr, QualType CType) {
  return HLSLBufferCopyEmitter(CGF, DestPtr, SrcPtr).emitCopy(CType);
}

LValue CGHLSLRuntime::emitBufferMemberExpr(CodeGenFunction &CGF,
                                           const MemberExpr *E) {
  LValue Base =
      CGF.EmitCheckedLValue(E->getBase(), CodeGenFunction::TCK_MemberAccess);
  auto *Field = dyn_cast<FieldDecl>(E->getMemberDecl());
  assert(Field && "Unexpected access into HLSL buffer");

  // Get the field index for the struct.
  const RecordDecl *Rec = Field->getParent();
  unsigned FieldIdx =
      CGM.getTypes().getCGRecordLayout(Rec).getLLVMFieldNo(Field);

  // Work out the buffer layout type to index into.
  QualType RecType = CGM.getContext().getCanonicalTagType(Rec);
  assert(RecType->isStructureOrClassType() && "Invalid type in HLSL buffer");
  // Since this is a member of an object in the buffer and not the buffer's
  // struct/class itself, we shouldn't have any offsets on the members we need
  // to contend with.
  CGHLSLOffsetInfo EmptyOffsets;
  llvm::StructType *LayoutTy = HLSLBufferLayoutBuilder(CGM).layOutStruct(
      RecType->getAsCanonical<RecordType>(), EmptyOffsets);

  // Now index into the struct, making sure that the type we return is the
  // buffer layout type rather than the original type in the AST.
  QualType FieldType = Field->getType();
  llvm::Type *FieldLLVMTy = CGM.getTypes().ConvertTypeForMem(FieldType);
  CharUnits Align = CharUnits::fromQuantity(
      CGF.CGM.getDataLayout().getABITypeAlign(FieldLLVMTy));
  Address Addr(CGF.Builder.CreateStructGEP(LayoutTy, Base.getPointer(CGF),
                                           FieldIdx, Field->getName()),
               FieldLLVMTy, Align, KnownNonNull);

  LValue LV = LValue::MakeAddr(Addr, FieldType, CGM.getContext(),
                               LValueBaseInfo(AlignmentSource::Type),
                               CGM.getTBAAAccessInfo(FieldType));
  LV.getQuals().addCVRQualifiers(Base.getVRQualifiers());

  return LV;
}
