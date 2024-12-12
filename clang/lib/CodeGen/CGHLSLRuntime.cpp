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

// Creates the LLVM struct type representing the shape of the constant buffer,
// which will be included in the LLVM target type, and calculates the memory
// layout and constant buffer layout offsets of each constant.
static void layoutBuffer(CGHLSLRuntime::Buffer &Buf, const DataLayout &DL) {
  assert(!Buf.Constants.empty() &&
         "empty constant buffer should not be created");

  std::vector<llvm::Type *> EltTys;
  unsigned MemOffset = 0, CBufOffset = 0, Size = 0;

  for (auto &C : Buf.Constants) {
    GlobalVariable *GV = C.GlobalVar;
    llvm::Type *Ty = GV->getValueType();

    assert(!Ty->isArrayTy() && !Ty->isStructTy() &&
           "arrays and structs in cbuffer are not yet implemened");

    // scalar type, vector or matrix
    EltTys.emplace_back(Ty);
    unsigned FieldSize = Ty->getScalarSizeInBits() / 8;
    if (Ty->isVectorTy())
      FieldSize *= cast<FixedVectorType>(Ty)->getNumElements();
    assert(FieldSize <= 16 && "field side larger than constant buffer row");

    // set memory layout offset (no padding)
    C.MemOffset = MemOffset;
    MemOffset += FieldSize;

    // calculate cbuffer layout offset or update total cbuffer size from
    // packoffset annotations
    if (Buf.HasPackoffset) {
      assert(C.CBufferOffset != UINT_MAX &&
             "cbuffer offset should have been set from packoffset attribute");
      unsigned OffsetAfterField = C.CBufferOffset + FieldSize;
      if (Size < OffsetAfterField)
        Size = OffsetAfterField;
    } else {
      // allign to the size of the field
      CBufOffset = llvm::alignTo(CBufOffset, FieldSize);
      C.CBufferOffset = CBufOffset;
      CBufOffset += FieldSize;
      Size = CBufOffset;
    }
  }
  Buf.LayoutStruct = llvm::StructType::get(EltTys[0]->getContext(), EltTys);
  Buf.Size = Size;
}

// Creates LLVM target type target("dx.CBuffer",..) for the constant buffer.
// The target type includes the LLVM struct type representing the shape
// of the constant buffer, size, and a list of offsets for each fields
// in cbuffer layout.
static llvm::Type *getBufferTargetType(LLVMContext &Ctx,
                                       CGHLSLRuntime::Buffer &Buf) {
  assert(Buf.LayoutStruct != nullptr && Buf.Size != UINT_MAX &&
         "the buffer layout has not been calculated yet");
  llvm::SmallVector<unsigned> SizeAndOffsets;
  SizeAndOffsets.reserve(Buf.Constants.size() + 1);
  SizeAndOffsets.push_back(Buf.Size);
  for (CGHLSLRuntime::BufferConstant &C : Buf.Constants) {
    SizeAndOffsets.push_back(C.CBufferOffset);
  }
  return llvm::TargetExtType::get(Ctx, "dx.CBuffer", {Buf.LayoutStruct},
                                  SizeAndOffsets);
}

// Replaces all uses of the temporary constant buffer global variables with
// buffer access intrinsic resource.getpointer.
static void replaceBufferGlobals(CodeGenModule &CGM,
                                 CGHLSLRuntime::Buffer &Buf) {
  assert(Buf.IsCBuffer && "tbuffer codegen is not yet supported");

  GlobalVariable *BufGV = Buf.GlobalVar;
  for (auto &Constant : Buf.Constants) {
    GlobalVariable *ConstGV = Constant.GlobalVar;

    // TODO: Map to an hlsl_device address space.
    llvm::Type *RetTy = ConstGV->getType();
    llvm::Type *TargetTy = BufGV->getValueType();

    // Replace all uses of GV with CBuffer access
    while (ConstGV->use_begin() != ConstGV->use_end()) {
      Use &U = *ConstGV->use_begin();
      if (Instruction *UserInstr = dyn_cast<Instruction>(U.getUser())) {
        IRBuilder<> Builder(UserInstr);
        Value *Handle = Builder.CreateLoad(TargetTy, BufGV);
        Value *ResGetPointer = Builder.CreateIntrinsic(
            RetTy, Intrinsic::dx_resource_getpointer,
            ArrayRef<llvm::Value *>{Handle,
                                    Builder.getInt32(Constant.MemOffset)});
        U.set(ResGetPointer);
      } else {
        llvm_unreachable("unexpected use of constant value");
      }
    }
    ConstGV->removeDeadConstantUsers();
    ConstGV->eraseFromParent();
  }
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

  assert(!D->getType()->isArrayType() && !D->getType()->isStructureType() &&
         "codegen for arrays and structs in cbuffer is not yet supported");

  auto *GV = cast<GlobalVariable>(CGM.GetAddrOfGlobalVar(D));
  // Add debug info for constVal.
  if (CGDebugInfo *DI = CGM.getModuleDebugInfo())
    if (CGM.getCodeGenOpts().getDebugInfo() >=
        codegenoptions::DebugInfoKind::LimitedDebugInfo)
      DI->EmitGlobalVariable(cast<GlobalVariable>(GV), D);

  CB.Constants.emplace_back(GV);

  if (HLSLPackOffsetAttr *PO = D->getAttr<HLSLPackOffsetAttr>()) {
    CB.HasPackoffset = true;
    CB.Constants.back().CBufferOffset = PO->getOffsetInBytes();
  }
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

// Creates temporary global variables for all declarations within the constant
// buffer context, calculates the buffer layouts, and then creates a global
// variable for the constant buffer and adds it to the module.
// All uses of the temporary constant globals will be replaced with buffer
// access intrinsic resource.getpointer in CGHLSLRuntime::finishCodeGen.
// Later on in DXILResourceAccess pass these will be transtaled
// to dx.op.cbufferLoadLegacy instructions.
void CGHLSLRuntime::addBuffer(const HLSLBufferDecl *D) {
  llvm::Module &M = CGM.getModule();
  const DataLayout &DL = M.getDataLayout();

  assert(D->isCBuffer() && "tbuffer codegen is not supported yet");

  Buffer &Buf = Buffers.emplace_back(D);
  addBufferDecls(D, Buf);
  if (Buf.Constants.empty()) {
    // empty constant buffer - do not add to globals
    Buffers.pop_back();
    return;
  }
  layoutBuffer(Buf, DL);

  // Create global variable for CB.
  llvm::Type *TargetTy = getBufferTargetType(CGM.getLLVMContext(), Buf);
  Buf.GlobalVar = new GlobalVariable(
      TargetTy, /*isConstant*/ true, GlobalValue::LinkageTypes::ExternalLinkage,
      nullptr, llvm::formatv("{0}{1}", Buf.Name, Buf.IsCBuffer ? ".cb" : ".tb"),
      GlobalValue::NotThreadLocal);

  M.insertGlobalVariable(Buf.GlobalVar);
  ResourcesToBind.emplace_back(Buf.Decl, Buf.GlobalVar);
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

  for (auto &Buf : Buffers) {
    replaceBufferGlobals(CGM, Buf);
  }
}

CGHLSLRuntime::Buffer::Buffer(const HLSLBufferDecl *D)
    : Name(D->getName()), IsCBuffer(D->isCBuffer()), HasPackoffset(false),
      LayoutStruct(nullptr), Decl(D), GlobalVar(nullptr) {}

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
      GV, RK, ET, IsROV, Binding.Slot.value_or(UINT_MAX), Binding.Space);
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
    Slot = Binding->getSlotNumber();
    Space = Binding->getSpaceNumber();
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
    for (auto *Fn : CtorFns)
      B.CreateCall(FunctionCallee(Fn), {}, OB);

    // Insert global dtors before the terminator of the last instruction
    B.SetInsertPoint(F.back().getTerminator());
    for (auto *Fn : DtorFns)
      B.CreateCall(FunctionCallee(Fn), {}, OB);
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

  for (const auto &[Decl, GV] : ResourcesToBind) {
    for (Attr *A : Decl->getAttrs()) {
      HLSLResourceBindingAttr *RBA = dyn_cast<HLSLResourceBindingAttr>(A);
      if (!RBA)
        continue;

      llvm::Type *TargetTy = nullptr;
      if (const VarDecl *VD = dyn_cast<VarDecl>(Decl)) {
        const HLSLAttributedResourceType *AttrResType =
            HLSLAttributedResourceType::findHandleTypeOnResource(
                VD->getType().getTypePtr());

        // FIXME: Only simple declarations of resources are supported for now.
        // Arrays of resources or resources in user defined classes are
        // not implemented yet.
        assert(
            AttrResType != nullptr &&
            "Resource class must have a handle of HLSLAttributedResourceType");

        TargetTy = CGM.getTargetCodeGenInfo().getHLSLType(CGM, AttrResType);
      } else {
        assert(isa<HLSLBufferDecl>(Decl));
        TargetTy = GV->getValueType();
      }
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
          nullptr, Twine(Decl->getName()).concat("_h"));

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
