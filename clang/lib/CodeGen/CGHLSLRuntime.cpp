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
#include "CodeGenModule.h"
#include "TargetInfo.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/TargetOptions.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Alignment.h"

#include "llvm/Support/FormatVariadic.h"

using namespace clang;
using namespace CodeGen;
using namespace clang::hlsl;
using namespace llvm;

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
void addDisableOptimizations(llvm::Module &M) {
  StringRef Key = "dx.disable_optimizations";
  M.addModuleFlag(llvm::Module::ModFlagBehavior::Override, Key, 1);
}
// cbuffer will be translated into global variable in special address space.
// If translate into C,
// cbuffer A {
//   float a;
//   float b;
// }
// float foo() { return a + b; }
//
// will be translated into
//
// struct A {
//   float a;
//   float b;
// } cbuffer_A __attribute__((address_space(4)));
// float foo() { return cbuffer_A.a + cbuffer_A.b; }
//
// layoutBuffer will create the struct A type.
// replaceBuffer will replace use of global variable a and b with cbuffer_A.a
// and cbuffer_A.b.
//
void layoutBuffer(CGHLSLRuntime::Buffer &Buf, const DataLayout &DL) {
  if (Buf.Constants.empty())
    return;

  std::vector<llvm::Type *> EltTys;
  for (auto &Const : Buf.Constants) {
    GlobalVariable *GV = Const.first;
    Const.second = EltTys.size();
    llvm::Type *Ty = GV->getValueType();
    EltTys.emplace_back(Ty);
  }
  Buf.LayoutStruct = llvm::StructType::get(EltTys[0]->getContext(), EltTys);
}

GlobalVariable *replaceBuffer(CGHLSLRuntime::Buffer &Buf) {
  // Create global variable for CB.
  GlobalVariable *CBGV = new GlobalVariable(
      Buf.LayoutStruct, /*isConstant*/ true,
      GlobalValue::LinkageTypes::ExternalLinkage, nullptr,
      llvm::formatv("{0}{1}", Buf.Name, Buf.IsCBuffer ? ".cb." : ".tb."),
      GlobalValue::NotThreadLocal);

  IRBuilder<> B(CBGV->getContext());
  Value *ZeroIdx = B.getInt32(0);
  // Replace Const use with CB use.
  for (auto &[GV, Offset] : Buf.Constants) {
    Value *GEP =
        B.CreateGEP(Buf.LayoutStruct, CBGV, {ZeroIdx, B.getInt32(Offset)});

    assert(Buf.LayoutStruct->getElementType(Offset) == GV->getValueType() &&
           "constant type mismatch");

    // Replace.
    GV->replaceAllUsesWith(GEP);
    // Erase GV.
    GV->removeDeadConstantUsers();
    GV->eraseFromParent();
  }
  return CBGV;
}

void addRootSignature(
    ArrayRef<llvm::hlsl::root_signature::RootElement> Elements,
    llvm::Function *Fn, llvm::Module &M) {
  auto &Ctx = M.getContext();

  llvm::hlsl::root_signature::MetadataBuilder Builder(Ctx, Elements);
  MDNode *RootSignature = Builder.BuildRootSignature();
  MDNode *FnPairing = MDNode::get(Ctx, {ValueAsMetadata::get(Fn),
                                        RootSignature});

  StringRef RootSignatureValKey = "dx.rootsignatures";
  auto *RootSignatureValMD = M.getOrInsertNamedMetadata(RootSignatureValKey);
  RootSignatureValMD->addOperand(FnPairing);
}

} // namespace

llvm::Type *CGHLSLRuntime::convertHLSLSpecificType(const Type *T) {
  assert(T->isHLSLSpecificType() && "Not an HLSL specific type!");

  // Check if the target has a specific translation for this type first.
  if (llvm::Type *TargetTy = CGM.getTargetCodeGenInfo().getHLSLType(CGM, T))
    return TargetTy;

  llvm_unreachable("Generic handling of HLSL types is not supported.");
}

llvm::Triple::ArchType CGHLSLRuntime::getArch() {
  return CGM.getTarget().getTriple().getArch();
}

void CGHLSLRuntime::addConstant(VarDecl *D, Buffer &CB) {
  if (D->getStorageClass() == SC_Static) {
    // For static inside cbuffer, take as global static.
    // Don't add to cbuffer.
    CGM.EmitGlobal(D);
    return;
  }

  auto *GV = cast<GlobalVariable>(CGM.GetAddrOfGlobalVar(D));
  // Add debug info for constVal.
  if (CGDebugInfo *DI = CGM.getModuleDebugInfo())
    if (CGM.getCodeGenOpts().getDebugInfo() >=
        codegenoptions::DebugInfoKind::LimitedDebugInfo)
      DI->EmitGlobalVariable(cast<GlobalVariable>(GV), D);

  // FIXME: support packoffset.
  // See https://github.com/llvm/llvm-project/issues/57914.
  uint32_t Offset = 0;
  bool HasUserOffset = false;

  unsigned LowerBound = HasUserOffset ? Offset : UINT_MAX;
  CB.Constants.emplace_back(std::make_pair(GV, LowerBound));
}

void CGHLSLRuntime::addBufferDecls(const DeclContext *DC, Buffer &CB) {
  for (Decl *it : DC->decls()) {
    if (auto *ConstDecl = dyn_cast<VarDecl>(it)) {
      addConstant(ConstDecl, CB);
    } else if (isa<CXXRecordDecl, EmptyDecl>(it)) {
      // Nothing to do for this declaration.
    } else if (isa<FunctionDecl>(it)) {
      // A function within an cbuffer is effectively a top-level function,
      // as it only refers to globally scoped declarations.
      CGM.EmitTopLevelDecl(it);
    }
  }
}

void CGHLSLRuntime::addBuffer(const HLSLBufferDecl *D) {
  Buffers.emplace_back(Buffer(D));
  addBufferDecls(D, Buffers.back());
}

void CGHLSLRuntime::finishCodeGen() {
  auto &TargetOpts = CGM.getTarget().getTargetOpts();
  llvm::Module &M = CGM.getModule();
  Triple T(M.getTargetTriple());
  if (T.getArch() == Triple::ArchType::dxil)
    addDxilValVersion(TargetOpts.DxilValidatorVersion, M);

  generateGlobalCtorDtorCalls();
  if (CGM.getCodeGenOpts().OptimizationLevel == 0)
    addDisableOptimizations(M);

  const DataLayout &DL = M.getDataLayout();

  for (auto &Buf : Buffers) {
    layoutBuffer(Buf, DL);
    GlobalVariable *GV = replaceBuffer(Buf);
    M.insertGlobalVariable(GV);
    llvm::hlsl::ResourceClass RC = Buf.IsCBuffer
                                       ? llvm::hlsl::ResourceClass::CBuffer
                                       : llvm::hlsl::ResourceClass::SRV;
    llvm::hlsl::ResourceKind RK = Buf.IsCBuffer
                                      ? llvm::hlsl::ResourceKind::CBuffer
                                      : llvm::hlsl::ResourceKind::TBuffer;
    addBufferResourceAnnotation(GV, RC, RK, /*IsROV=*/false,
                                llvm::hlsl::ElementType::Invalid, Buf.Binding);
  }
}

CGHLSLRuntime::Buffer::Buffer(const HLSLBufferDecl *D)
    : Name(D->getName()), IsCBuffer(D->isCBuffer()),
      Binding(D->getAttr<HLSLResourceBindingAttr>()) {}

void CGHLSLRuntime::addBufferResourceAnnotation(llvm::GlobalVariable *GV,
                                                llvm::hlsl::ResourceClass RC,
                                                llvm::hlsl::ResourceKind RK,
                                                bool IsROV,
                                                llvm::hlsl::ElementType ET,
                                                BufferResBinding &Binding) {
  llvm::Module &M = CGM.getModule();

  NamedMDNode *ResourceMD = nullptr;
  switch (RC) {
  case llvm::hlsl::ResourceClass::UAV:
    ResourceMD = M.getOrInsertNamedMetadata("hlsl.uavs");
    break;
  case llvm::hlsl::ResourceClass::SRV:
    ResourceMD = M.getOrInsertNamedMetadata("hlsl.srvs");
    break;
  case llvm::hlsl::ResourceClass::CBuffer:
    ResourceMD = M.getOrInsertNamedMetadata("hlsl.cbufs");
    break;
  default:
    assert(false && "Unsupported buffer type!");
    return;
  }
  assert(ResourceMD != nullptr &&
         "ResourceMD must have been set by the switch above.");

  llvm::hlsl::FrontendResource Res(
      GV, RK, ET, IsROV, Binding.Reg.value_or(UINT_MAX), Binding.Space);
  ResourceMD->addOperand(Res.getMetadata());
}

static llvm::hlsl::ElementType
calculateElementType(const ASTContext &Context, const clang::Type *ResourceTy) {
  using llvm::hlsl::ElementType;

  // TODO: We may need to update this when we add things like ByteAddressBuffer
  // that don't have a template parameter (or, indeed, an element type).
  const auto *TST = ResourceTy->getAs<TemplateSpecializationType>();
  assert(TST && "Resource types must be template specializations");
  ArrayRef<TemplateArgument> Args = TST->template_arguments();
  assert(!Args.empty() && "Resource has no element type");

  // At this point we have a resource with an element type, so we can assume
  // that it's valid or we would have diagnosed the error earlier.
  QualType ElTy = Args[0].getAsType();

  // We should either have a basic type or a vector of a basic type.
  if (const auto *VecTy = ElTy->getAs<clang::VectorType>())
    ElTy = VecTy->getElementType();

  if (ElTy->isSignedIntegerType()) {
    switch (Context.getTypeSize(ElTy)) {
    case 16:
      return ElementType::I16;
    case 32:
      return ElementType::I32;
    case 64:
      return ElementType::I64;
    }
  } else if (ElTy->isUnsignedIntegerType()) {
    switch (Context.getTypeSize(ElTy)) {
    case 16:
      return ElementType::U16;
    case 32:
      return ElementType::U32;
    case 64:
      return ElementType::U64;
    }
  } else if (ElTy->isSpecificBuiltinType(BuiltinType::Half))
    return ElementType::F16;
  else if (ElTy->isSpecificBuiltinType(BuiltinType::Float))
    return ElementType::F32;
  else if (ElTy->isSpecificBuiltinType(BuiltinType::Double))
    return ElementType::F64;

  // TODO: We need to handle unorm/snorm float types here once we support them
  llvm_unreachable("Invalid element type for resource");
}

void CGHLSLRuntime::annotateHLSLResource(const VarDecl *D, GlobalVariable *GV) {
  const Type *Ty = D->getType()->getPointeeOrArrayElementType();
  if (!Ty)
    return;
  const auto *RD = Ty->getAsCXXRecordDecl();
  if (!RD)
    return;
  // the resource related attributes are on the handle member
  // inside the record decl
  for (auto *FD : RD->fields()) {
    const auto *HLSLResAttr = FD->getAttr<HLSLResourceAttr>();
    const HLSLAttributedResourceType *AttrResType =
        dyn_cast<HLSLAttributedResourceType>(FD->getType().getTypePtr());
    if (!HLSLResAttr || !AttrResType)
      continue;

    llvm::hlsl::ResourceClass RC = AttrResType->getAttrs().ResourceClass;
    if (RC == llvm::hlsl::ResourceClass::UAV ||
        RC == llvm::hlsl::ResourceClass::SRV)
      // UAVs and SRVs have already been converted to use LLVM target types,
      // we can disable generating of these resource annotations. This will
      // enable progress on structured buffers with user defined types this
      // resource annotations code does not handle and it crashes.
      // This whole function is going to be removed as soon as cbuffers are
      // converted to target types (llvm/llvm-project #114126).
      return;

    bool IsROV = AttrResType->getAttrs().IsROV;
    llvm::hlsl::ResourceKind RK = HLSLResAttr->getResourceKind();
    llvm::hlsl::ElementType ET = calculateElementType(CGM.getContext(), Ty);

    BufferResBinding Binding(D->getAttr<HLSLResourceBindingAttr>());
    addBufferResourceAnnotation(GV, RC, RK, IsROV, ET, Binding);
  }
}

CGHLSLRuntime::BufferResBinding::BufferResBinding(
    HLSLResourceBindingAttr *Binding) {
  if (Binding) {
    llvm::APInt RegInt(64, 0);
    Binding->getSlot().substr(1).getAsInteger(10, RegInt);
    Reg = RegInt.getLimitedValue();
    llvm::APInt SpaceInt(64, 0);
    Binding->getSpace().substr(5).getAsInteger(10, SpaceInt);
    Space = SpaceInt.getLimitedValue();
  } else {
    Space = 0;
  }
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

llvm::Value *CGHLSLRuntime::emitInputSemantic(IRBuilder<> &B,
                                              const ParmVarDecl &D,
                                              llvm::Type *Ty) {
  assert(D.hasAttrs() && "Entry parameter missing annotation attribute!");
  if (D.hasAttr<HLSLSV_GroupIndexAttr>()) {
    llvm::Function *DxGroupIndex =
        CGM.getIntrinsic(Intrinsic::dx_flattened_thread_id_in_group);
    return B.CreateCall(FunctionCallee(DxGroupIndex));
  }
  if (D.hasAttr<HLSLSV_DispatchThreadIDAttr>()) {
    llvm::Function *ThreadIDIntrinsic =
        CGM.getIntrinsic(getThreadIdIntrinsic());
    return buildVectorInput(B, ThreadIDIntrinsic, Ty);
  }
  if (D.hasAttr<HLSLSV_GroupThreadIDAttr>()) {
    llvm::Function *GroupThreadIDIntrinsic =
        CGM.getIntrinsic(getGroupThreadIdIntrinsic());
    return buildVectorInput(B, GroupThreadIDIntrinsic, Ty);
  }
  if (D.hasAttr<HLSLSV_GroupIDAttr>()) {
    llvm::Function *GroupIDIntrinsic = CGM.getIntrinsic(Intrinsic::dx_group_id);
    return buildVectorInput(B, GroupIDIntrinsic, Ty);
  }
  assert(false && "Unhandled parameter attribute");
  return nullptr;
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
    llvm::Value *I = B.CreateIntrinsic(
        llvm::Intrinsic::experimental_convergence_entry, {}, {});
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
    Args.push_back(emitInputSemantic(B, *PD, Param.getType()));
  }

  CallInst *CI = B.CreateCall(FunctionCallee(Fn), Args, OB);
  CI->setCallingConv(Fn->getCallingConv());
  // FIXME: Handle codegen for return type semantics.
  // See: https://github.com/llvm/llvm-project/issues/57875
  B.CreateRetVoid();

  // Add and identify root signature to function, if applicable
  const AttrVec &Attrs = FD->getAttrs();
  for (const Attr *Attr : Attrs) {
    if (const auto *RSAttr = dyn_cast<HLSLRootSignatureAttr>(Attr))
      addRootSignature(RSAttr->getElements(), EntryFn, M);
  }
}

void CGHLSLRuntime::setHLSLFunctionAttributes(const FunctionDecl *FD,
                                              llvm::Function *Fn) {
  if (FD->isInExportDeclContext()) {
    const StringRef ExportAttrKindStr = "hlsl.export";
    Fn->addFnAttr(ExportAttrKindStr);
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

  llvm::SmallVector<Function *> CtorFns;
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

void CGHLSLRuntime::handleGlobalVarDefinition(const VarDecl *VD,
                                              llvm::GlobalVariable *GV) {
  // If the global variable has resource binding, add it to the list of globals
  // that need resource binding initialization.
  const HLSLResourceBindingAttr *RBA = VD->getAttr<HLSLResourceBindingAttr>();
  if (!RBA)
    return;

  if (!HLSLAttributedResourceType::findHandleTypeOnResource(
          VD->getType().getTypePtr()))
    // FIXME: Only simple declarations of resources are supported for now.
    // Arrays of resources or resources in user defined classes are
    // not implemented yet.
    return;

  ResourcesToBind.emplace_back(VD, GV);
}

bool CGHLSLRuntime::needsResourceBindingInitFn() {
  return !ResourcesToBind.empty();
}

llvm::Function *CGHLSLRuntime::createResourceBindingInitFn() {
  // No resources to bind
  assert(needsResourceBindingInitFn() && "no resources to bind");

  LLVMContext &Ctx = CGM.getLLVMContext();
  llvm::Type *Int1Ty = llvm::Type::getInt1Ty(Ctx);

  llvm::Function *InitResBindingsFunc =
      llvm::Function::Create(llvm::FunctionType::get(CGM.VoidTy, false),
                             llvm::GlobalValue::InternalLinkage,
                             "_init_resource_bindings", CGM.getModule());

  llvm::BasicBlock *EntryBB =
      llvm::BasicBlock::Create(Ctx, "entry", InitResBindingsFunc);
  CGBuilderTy Builder(CGM, Ctx);
  const DataLayout &DL = CGM.getModule().getDataLayout();
  Builder.SetInsertPoint(EntryBB);

  for (const auto &[VD, GV] : ResourcesToBind) {
    for (Attr *A : VD->getAttrs()) {
      HLSLResourceBindingAttr *RBA = dyn_cast<HLSLResourceBindingAttr>(A);
      if (!RBA)
        continue;

      const HLSLAttributedResourceType *AttrResType =
          HLSLAttributedResourceType::findHandleTypeOnResource(
              VD->getType().getTypePtr());

      // FIXME: Only simple declarations of resources are supported for now.
      // Arrays of resources or resources in user defined classes are
      // not implemented yet.
      assert(AttrResType != nullptr &&
             "Resource class must have a handle of HLSLAttributedResourceType");

      llvm::Type *TargetTy =
          CGM.getTargetCodeGenInfo().getHLSLType(CGM, AttrResType);
      assert(TargetTy != nullptr &&
             "Failed to convert resource handle to target type");

      auto *Space = llvm::ConstantInt::get(CGM.IntTy, RBA->getSpaceNumber());
      auto *Slot = llvm::ConstantInt::get(CGM.IntTy, RBA->getSlotNumber());
      // FIXME: resource arrays are not yet implemented
      auto *Range = llvm::ConstantInt::get(CGM.IntTy, 1);
      auto *Index = llvm::ConstantInt::get(CGM.IntTy, 0);
      // FIXME: NonUniformResourceIndex bit is not yet implemented
      auto *NonUniform = llvm::ConstantInt::get(Int1Ty, false);
      llvm::Value *Args[] = {Space, Slot, Range, Index, NonUniform};

      llvm::Value *CreateHandle = Builder.CreateIntrinsic(
          /*ReturnType=*/TargetTy, getCreateHandleFromBindingIntrinsic(), Args,
          nullptr, Twine(VD->getName()).concat("_h"));

      llvm::Value *HandleRef =
          Builder.CreateStructGEP(GV->getValueType(), GV, 0);
      Builder.CreateAlignedStore(CreateHandle, HandleRef,
                                 HandleRef->getPointerAlignment(DL));
    }
  }

  Builder.CreateRetVoid();
  return InitResBindingsFunc;
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
