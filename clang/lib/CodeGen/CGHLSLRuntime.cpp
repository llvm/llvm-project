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
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "TargetInfo.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attrs.inc"
#include "clang/AST/Decl.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Type.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Frontend/FrontendDiagnostic.h"
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

void addRootSignature(llvm::dxbc::RootSignatureVersion RootSigVer,
                      ArrayRef<llvm::hlsl::rootsig::RootElement> Elements,
                      llvm::Function *Fn, llvm::Module &M) {
  auto &Ctx = M.getContext();

  llvm::hlsl::rootsig::MetadataBuilder RSBuilder(Ctx, Elements);
  MDNode *RootSignature = RSBuilder.BuildRootSignature();

  ConstantAsMetadata *Version = ConstantAsMetadata::get(ConstantInt::get(
      llvm::Type::getInt32Ty(Ctx), llvm::to_underlying(RootSigVer)));
  MDNode *MDVals =
      MDNode::get(Ctx, {ValueAsMetadata::get(Fn), RootSignature, Version});

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

// Find constructor decl for a specific resource record type and binding
// (implicit vs. explicit). The constructor has 5 parameters.
// For explicit binding the signature is:
//   void(unsigned, unsigned, int, unsigned, const char *).
// For implicit binding the signature is:
//   void(unsigned, int, unsigned, unsigned, const char *).
static CXXConstructorDecl *findResourceConstructorDecl(ASTContext &AST,
                                                       QualType ResTy,
                                                       bool ExplicitBinding) {
  std::array<QualType, 5> ExpParmTypes = {
      AST.UnsignedIntTy, AST.UnsignedIntTy, AST.UnsignedIntTy,
      AST.UnsignedIntTy, AST.getPointerType(AST.CharTy.withConst())};
  ExpParmTypes[ExplicitBinding ? 2 : 1] = AST.IntTy;

  CXXRecordDecl *ResDecl = ResTy->getAsCXXRecordDecl();
  for (auto *Ctor : ResDecl->ctors()) {
    if (Ctor->getNumParams() != ExpParmTypes.size())
      continue;
    auto *ParmIt = Ctor->param_begin();
    auto ExpTyIt = ExpParmTypes.begin();
    for (; ParmIt != Ctor->param_end() && ExpTyIt != ExpParmTypes.end();
         ++ParmIt, ++ExpTyIt) {
      if ((*ParmIt)->getType() != *ExpTyIt)
        break;
    }
    if (ParmIt == Ctor->param_end())
      return Ctor;
  }
  llvm_unreachable("did not find constructor for resource class");
}

static Value *buildNameForResource(llvm::StringRef BaseName,
                                   CodeGenModule &CGM) {
  llvm::SmallString<64> GlobalName = {BaseName, ".str"};
  return CGM.GetAddrOfConstantCString(BaseName.str(), GlobalName.c_str())
      .getPointer();
}

static void createResourceCtorArgs(CodeGenModule &CGM, CXXConstructorDecl *CD,
                                   llvm::Value *ThisPtr, llvm::Value *Range,
                                   llvm::Value *Index, StringRef Name,
                                   HLSLResourceBindingAttr *RBA,
                                   HLSLVkBindingAttr *VkBinding,
                                   CallArgList &Args) {
  assert((VkBinding || RBA) && "at least one a binding attribute expected");

  std::optional<uint32_t> RegisterSlot;
  uint32_t SpaceNo = 0;
  if (VkBinding) {
    RegisterSlot = VkBinding->getBinding();
    SpaceNo = VkBinding->getSet();
  } else {
    if (RBA->hasRegisterSlot())
      RegisterSlot = RBA->getSlotNumber();
    SpaceNo = RBA->getSpaceNumber();
  }

  ASTContext &AST = CD->getASTContext();
  Value *NameStr = buildNameForResource(Name, CGM);
  Value *Space = llvm::ConstantInt::get(CGM.IntTy, SpaceNo);

  Args.add(RValue::get(ThisPtr), CD->getThisType());
  if (RegisterSlot.has_value()) {
    // explicit binding
    auto *RegSlot = llvm::ConstantInt::get(CGM.IntTy, RegisterSlot.value());
    Args.add(RValue::get(RegSlot), AST.UnsignedIntTy);
    Args.add(RValue::get(Space), AST.UnsignedIntTy);
    Args.add(RValue::get(Range), AST.IntTy);
    Args.add(RValue::get(Index), AST.UnsignedIntTy);

  } else {
    // implicit binding
    assert(RBA && "missing implicit binding attribute");
    auto *OrderID =
        llvm::ConstantInt::get(CGM.IntTy, RBA->getImplicitBindingOrderID());
    Args.add(RValue::get(Space), AST.UnsignedIntTy);
    Args.add(RValue::get(Range), AST.IntTy);
    Args.add(RValue::get(Index), AST.UnsignedIntTy);
    Args.add(RValue::get(OrderID), AST.UnsignedIntTy);
  }
  Args.add(RValue::get(NameStr), AST.getPointerType(AST.CharTy.withConst()));
}

// Initializes local resource array variable. For multi-dimensional arrays it
// calls itself recursively to initialize its sub-arrays. The Index used in the
// resource constructor calls will begin at StartIndex and will be incremented
// for each array element. The last used resource Index is returned to the
// caller.
static Value *initializeLocalResourceArray(
    CodeGenFunction &CGF, AggValueSlot &ValueSlot,
    const ConstantArrayType *ArrayTy, CXXConstructorDecl *CD,
    llvm::Value *Range, llvm::Value *StartIndex, StringRef ResourceName,
    HLSLResourceBindingAttr *RBA, HLSLVkBindingAttr *VkBinding,
    ArrayRef<llvm::Value *> PrevGEPIndices, SourceLocation ArraySubsExprLoc) {

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
      Index = initializeLocalResourceArray(
          CGF, ValueSlot, SubArrayTy, CD, Range, Index, ResourceName, RBA,
          VkBinding, GEPIndices, ArraySubsExprLoc);
    }
    return Index;
  }

  // For array of resources, initialize each resource in the array.
  llvm::Type *Ty = CGF.ConvertTypeForMem(ElemType);
  CharUnits ElemSize = CD->getASTContext().getTypeSizeInChars(ElemType);
  CharUnits Align =
      TmpArrayAddr.getAlignment().alignmentOfArrayElement(ElemSize);

  for (uint64_t I = 0; I < ArraySize; I++) {
    if (I > 0) {
      Index = CGF.Builder.CreateAdd(Index, One);
      GEPIndices.back() = llvm::ConstantInt::get(IntTy, I);
    }
    Address ThisAddress =
        CGF.Builder.CreateGEP(TmpArrayAddr, GEPIndices, Ty, Align);
    llvm::Value *ThisPtr = CGF.getAsNaturalPointerTo(ThisAddress, ElemType);

    CallArgList Args;
    createResourceCtorArgs(CGF.CGM, CD, ThisPtr, Range, Index, ResourceName,
                           RBA, VkBinding, Args);
    CGF.EmitCXXConstructorCall(CD, Ctor_Complete, false, false, ThisAddress,
                               Args, ValueSlot.mayOverlap(), ArraySubsExprLoc,
                               ValueSlot.isSanitizerChecked());
  }
  return Index;
}

} // namespace

llvm::Type *
CGHLSLRuntime::convertHLSLSpecificType(const Type *T,
                                       SmallVector<int32_t> *Packoffsets) {
  assert(T->isHLSLSpecificType() && "Not an HLSL specific type!");

  // Check if the target has a specific translation for this type first.
  if (llvm::Type *TargetTy =
          CGM.getTargetCodeGenInfo().getHLSLType(CGM, T, Packoffsets))
    return TargetTy;

  llvm_unreachable("Generic handling of HLSL types is not supported.");
}

llvm::Triple::ArchType CGHLSLRuntime::getArch() {
  return CGM.getTarget().getTriple().getArch();
}

// Emits constant global variables for buffer constants declarations
// and creates metadata linking the constant globals with the buffer global.
void CGHLSLRuntime::emitBufferGlobalsAndMetadata(const HLSLBufferDecl *BufDecl,
                                                 llvm::GlobalVariable *BufGV) {
  LLVMContext &Ctx = CGM.getLLVMContext();

  // get the layout struct from constant buffer target type
  llvm::Type *BufType = BufGV->getValueType();
  llvm::Type *BufLayoutType =
      cast<llvm::TargetExtType>(BufType)->getTypeParameter(0);
  llvm::StructType *LayoutStruct = cast<llvm::StructType>(
      cast<llvm::TargetExtType>(BufLayoutType)->getTypeParameter(0));

  // Start metadata list associating the buffer global variable with its
  // constatns
  SmallVector<llvm::Metadata *> BufGlobals;
  BufGlobals.push_back(ValueAsMetadata::get(BufGV));

  const auto *ElemIt = LayoutStruct->element_begin();
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

    assert(ElemIt != LayoutStruct->element_end() &&
           "number of elements in layout struct does not match");
    llvm::Type *LayoutType = *ElemIt++;

    // FIXME: handle resources inside user defined structs
    // (llvm/wg-hlsl#175)

    // create global variable for the constant and to metadata list
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

// Iterates over all declarations in the HLSL buffer and based on the
// packoffset or register(c#) annotations it fills outs the Layout
// vector with the user-specified layout offsets.
// The buffer offsets can be specified 2 ways:
// 1. declarations in cbuffer {} block can have a packoffset annotation
//    (translates to HLSLPackOffsetAttr)
// 2. default constant buffer declarations at global scope can have
//    register(c#) annotations (translates to HLSLResourceBindingAttr with
//    RegisterType::C)
// It is not guaranteed that all declarations in a buffer have an annotation.
// For those where it is not specified a -1 value is added to the Layout
// vector. In the final layout these declarations will be placed at the end
// of the HLSL buffer after all of the elements with specified offset.
static void fillPackoffsetLayout(const HLSLBufferDecl *BufDecl,
                                 SmallVector<int32_t> &Layout) {
  assert(Layout.empty() && "expected empty vector for layout");
  assert(BufDecl->hasValidPackoffset());

  for (Decl *D : BufDecl->buffer_decls()) {
    if (isa<CXXRecordDecl, EmptyDecl>(D) || isa<FunctionDecl>(D)) {
      continue;
    }
    VarDecl *VD = dyn_cast<VarDecl>(D);
    if (!VD || VD->getType().getAddressSpace() != LangAS::hlsl_constant)
      continue;

    if (!VD->hasAttrs()) {
      Layout.push_back(-1);
      continue;
    }

    int32_t Offset = -1;
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
    Layout.push_back(Offset);
  }
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
  SmallVector<int32_t> Layout;
  if (BufDecl->hasValidPackoffset())
    fillPackoffsetLayout(BufDecl, Layout);

  llvm::TargetExtType *TargetTy =
      cast<llvm::TargetExtType>(convertHLSLSpecificType(
          ResHandleTy, BufDecl->hasValidPackoffset() ? &Layout : nullptr));
  llvm::GlobalVariable *BufGV = new GlobalVariable(
      TargetTy, /*isConstant*/ false,
      GlobalValue::LinkageTypes::ExternalLinkage, PoisonValue::get(TargetTy),
      llvm::formatv("{0}{1}", BufDecl->getName(),
                    BufDecl->isCBuffer() ? ".cb" : ".tb"),
      GlobalValue::NotThreadLocal);
  CGM.getModule().insertGlobalVariable(BufGV);

  // Add globals for constant buffer elements and create metadata nodes
  emitBufferGlobalsAndMetadata(BufDecl, BufGV);

  // Initialize cbuffer from binding (implicit or explicit)
  if (HLSLVkBindingAttr *VkBinding = BufDecl->getAttr<HLSLVkBindingAttr>()) {
    initializeBufferFromBinding(BufDecl, BufGV, VkBinding);
  } else {
    HLSLResourceBindingAttr *RBA = BufDecl->getAttr<HLSLResourceBindingAttr>();
    assert(RBA &&
           "cbuffer/tbuffer should always have resource binding attribute");
    initializeBufferFromBinding(BufDecl, BufGV, RBA);
  }
}

llvm::TargetExtType *
CGHLSLRuntime::getHLSLBufferLayoutType(const RecordType *StructType) {
  const auto Entry = LayoutTypes.find(StructType);
  if (Entry != LayoutTypes.end())
    return Entry->getSecond();
  return nullptr;
}

void CGHLSLRuntime::addHLSLBufferLayoutType(const RecordType *StructType,
                                            llvm::TargetExtType *LayoutTy) {
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

llvm::Value *
CGHLSLRuntime::emitSystemSemanticLoad(IRBuilder<> &B, llvm::Type *Type,
                                      const clang::DeclaratorDecl *Decl,
                                      SemanticInfo &ActiveSemantic) {
  if (isa<HLSLSV_GroupIndexAttr>(ActiveSemantic.Semantic)) {
    llvm::Function *GroupIndex =
        CGM.getIntrinsic(getFlattenedThreadIdInGroupIntrinsic());
    return B.CreateCall(FunctionCallee(GroupIndex));
  }

  if (isa<HLSLSV_DispatchThreadIDAttr>(ActiveSemantic.Semantic)) {
    llvm::Intrinsic::ID IntrinID = getThreadIdIntrinsic();
    llvm::Function *ThreadIDIntrinsic =
        llvm::Intrinsic::isOverloaded(IntrinID)
            ? CGM.getIntrinsic(IntrinID, {CGM.Int32Ty})
            : CGM.getIntrinsic(IntrinID);
    return buildVectorInput(B, ThreadIDIntrinsic, Type);
  }

  if (isa<HLSLSV_GroupThreadIDAttr>(ActiveSemantic.Semantic)) {
    llvm::Intrinsic::ID IntrinID = getGroupThreadIdIntrinsic();
    llvm::Function *GroupThreadIDIntrinsic =
        llvm::Intrinsic::isOverloaded(IntrinID)
            ? CGM.getIntrinsic(IntrinID, {CGM.Int32Ty})
            : CGM.getIntrinsic(IntrinID);
    return buildVectorInput(B, GroupThreadIDIntrinsic, Type);
  }

  if (isa<HLSLSV_GroupIDAttr>(ActiveSemantic.Semantic)) {
    llvm::Intrinsic::ID IntrinID = getGroupIdIntrinsic();
    llvm::Function *GroupIDIntrinsic =
        llvm::Intrinsic::isOverloaded(IntrinID)
            ? CGM.getIntrinsic(IntrinID, {CGM.Int32Ty})
            : CGM.getIntrinsic(IntrinID);
    return buildVectorInput(B, GroupIDIntrinsic, Type);
  }

  if (HLSLSV_PositionAttr *S =
          dyn_cast<HLSLSV_PositionAttr>(ActiveSemantic.Semantic)) {
    if (CGM.getTriple().getEnvironment() == Triple::EnvironmentType::Pixel)
      return createSPIRVBuiltinLoad(B, CGM.getModule(), Type,
                                    S->getAttrName()->getName(),
                                    /* BuiltIn::FragCoord */ 15);
  }

  llvm_unreachable("non-handled system semantic. FIXME.");
}

llvm::Value *
CGHLSLRuntime::handleScalarSemanticLoad(IRBuilder<> &B, llvm::Type *Type,
                                        const clang::DeclaratorDecl *Decl,
                                        SemanticInfo &ActiveSemantic) {

  if (!ActiveSemantic.Semantic) {
    ActiveSemantic.Semantic = Decl->getAttr<HLSLSemanticAttr>();
    if (!ActiveSemantic.Semantic) {
      CGM.getDiags().Report(Decl->getInnerLocStart(),
                            diag::err_hlsl_semantic_missing);
      return nullptr;
    }
    ActiveSemantic.Index = ActiveSemantic.Semantic->getSemanticIndex();
  }

  return emitSystemSemanticLoad(B, Type, Decl, ActiveSemantic);
}

llvm::Value *
CGHLSLRuntime::handleSemanticLoad(IRBuilder<> &B, llvm::Type *Type,
                                  const clang::DeclaratorDecl *Decl,
                                  SemanticInfo &ActiveSemantic) {
  assert(!Type->isStructTy());
  return handleScalarSemanticLoad(B, Type, Decl, ActiveSemantic);
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
    SemanticInfo ActiveSemantic = {nullptr, 0};
    Args.push_back(handleSemanticLoad(B, Param.getType(), PD, ActiveSemantic));
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
      addRootSignature(RSDecl->getVersion(), RSDecl->getRootElements(), EntryFn,
                       M);
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
                                                llvm::GlobalVariable *GV,
                                                HLSLVkBindingAttr *VkBinding) {
  assert(VkBinding && "expect a nonnull binding attribute");
  auto *Index = llvm::ConstantInt::get(CGM.IntTy, 0);
  auto *RangeSize = llvm::ConstantInt::get(CGM.IntTy, 1);
  auto *Set = llvm::ConstantInt::get(CGM.IntTy, VkBinding->getSet());
  auto *Binding = llvm::ConstantInt::get(CGM.IntTy, VkBinding->getBinding());
  Value *Name = buildNameForResource(BufDecl->getName(), CGM);
  llvm::Intrinsic::ID IntrinsicID =
      CGM.getHLSLRuntime().getCreateHandleFromBindingIntrinsic();

  SmallVector<Value *> Args{Set, Binding, RangeSize, Index, Name};
  initializeBuffer(CGM, GV, IntrinsicID, Args);
}

void CGHLSLRuntime::initializeBufferFromBinding(const HLSLBufferDecl *BufDecl,
                                                llvm::GlobalVariable *GV,
                                                HLSLResourceBindingAttr *RBA) {
  assert(RBA && "expect a nonnull binding attribute");
  auto *Index = llvm::ConstantInt::get(CGM.IntTy, 0);
  auto *RangeSize = llvm::ConstantInt::get(CGM.IntTy, 1);
  auto *Space = llvm::ConstantInt::get(CGM.IntTy, RBA->getSpaceNumber());
  Value *Name = buildNameForResource(BufDecl->getName(), CGM);

  llvm::Intrinsic::ID IntrinsicID =
      RBA->hasRegisterSlot()
          ? CGM.getHLSLRuntime().getCreateHandleFromBindingIntrinsic()
          : CGM.getHLSLRuntime().getCreateHandleFromImplicitBindingIntrinsic();

  // buffer with explicit binding
  if (RBA->hasRegisterSlot()) {
    auto *RegSlot = llvm::ConstantInt::get(CGM.IntTy, RBA->getSlotNumber());
    SmallVector<Value *> Args{Space, RegSlot, RangeSize, Index, Name};
    initializeBuffer(CGM, GV, IntrinsicID, Args);
  } else {
    // buffer with implicit binding
    auto *OrderID =
        llvm::ConstantInt::get(CGM.IntTy, RBA->getImplicitBindingOrderID());
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
  assert(ArraySubsExpr->getType()->isHLSLResourceRecord() ||
         ArraySubsExpr->getType()->isHLSLResourceRecordArray() &&
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
  HLSLVkBindingAttr *VkBinding = ArrayDecl->getAttr<HLSLVkBindingAttr>();
  HLSLResourceBindingAttr *RBA = ArrayDecl->getAttr<HLSLResourceBindingAttr>();
  assert((VkBinding || RBA) && "resource array must have a binding attribute");

  // Find the individual resource type.
  QualType ResultTy = ArraySubsExpr->getType();
  QualType ResourceTy =
      ResultTy->isArrayType() ? AST.getBaseElementType(ResultTy) : ResultTy;

  // Lookup the resource class constructor based on the resource type and
  // binding.
  CXXConstructorDecl *CD = findResourceConstructorDecl(
      AST, ResourceTy, VkBinding || RBA->hasRegisterSlot());

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
  Address TmpVarAddress = ValueSlot.getAddress();

  // Calculate total array size (= range size).
  llvm::Value *Range =
      llvm::ConstantInt::get(CGM.IntTy, getTotalArraySize(AST, ResArrayTy));

  // If the result of the subscript operation is a single resource, call the
  // constructor.
  if (ResultTy == ResourceTy) {
    QualType ThisType = CD->getThisType()->getPointeeType();
    llvm::Value *ThisPtr = CGF.getAsNaturalPointerTo(TmpVarAddress, ThisType);

    // Assemble the constructor parameters.
    CallArgList Args;
    createResourceCtorArgs(CGM, CD, ThisPtr, Range, Index, ArrayDecl->getName(),
                           RBA, VkBinding, Args);
    // Call the constructor.
    CGF.EmitCXXConstructorCall(CD, Ctor_Complete, false, false, TmpVarAddress,
                               Args, ValueSlot.mayOverlap(),
                               ArraySubsExpr->getExprLoc(),
                               ValueSlot.isSanitizerChecked());
  } else {
    // The result of the subscript operation is a local resource array which
    // needs to be initialized.
    const ConstantArrayType *ArrayTy =
        cast<ConstantArrayType>(ResultTy.getTypePtr());
    initializeLocalResourceArray(CGF, ValueSlot, ArrayTy, CD, Range, Index,
                                 ArrayDecl->getName(), RBA, VkBinding,
                                 {llvm::ConstantInt::get(CGM.IntTy, 0)},
                                 ArraySubsExpr->getExprLoc());
  }
  return CGF.MakeAddrLValue(TmpVar, ResultTy, AlignmentSource::Decl);
}
