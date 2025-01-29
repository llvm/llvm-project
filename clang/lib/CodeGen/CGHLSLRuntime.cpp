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
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Type.h"
#include "clang/Basic/TargetOptions.h"
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

using namespace clang;
using namespace CodeGen;
using namespace clang::hlsl;
using namespace llvm;

static void createResourceInitFn(CodeGenModule &CGM, llvm::GlobalVariable *GV,
                                 unsigned Slot, unsigned Space);

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

// Returns true if the type is an HLSL resource class
static bool isResourceRecordType(const clang::Type *Ty) {
  return HLSLAttributedResourceType::findHandleTypeOnResource(Ty) != nullptr;
}

// Returns true if the type is an HLSL resource class or an array of them
static bool isResourceRecordTypeOrArrayOf(const clang::Type *Ty) {
  while (const ConstantArrayType *CAT = dyn_cast<ConstantArrayType>(Ty))
    Ty = CAT->getArrayElementTypeNoTypeQual();
  return isResourceRecordType(Ty);
}

static ConstantAsMetadata *getConstIntMetadata(LLVMContext &Ctx, uint32_t value,
                                               bool isSigned = false) {
  return ConstantAsMetadata::get(
      ConstantInt::get(Ctx, llvm::APInt(32, value, isSigned)));
}

static unsigned getScalarOrVectorSize(llvm::Type *Ty) {
  assert(Ty->isVectorTy() || Ty->isIntegerTy() || Ty->isFloatingPointTy());
  if (Ty->isVectorTy()) {
    llvm::FixedVectorType *FVT = cast<llvm::FixedVectorType>(Ty);
    return FVT->getNumElements() *
           (FVT->getElementType()->getScalarSizeInBits() / 8);
  }
  return Ty->getScalarSizeInBits() / 8;
}

// Returns size of a struct in constant buffer layout. The sizes are cached
// in StructSizesForBuffer map. The map is also an indicator if a layout
// metadata for this struct has been added to the module.
// If the struct type is not in the map, this method will calculate the struct
// layout, add a metadata node describing it to the module, and add the struct
// size to the map.
size_t
CGHLSLRuntime::getOrCalculateStructSizeForBuffer(llvm::StructType *StructTy) {
  // check if we already have a side for this struct
  auto SizeIt = StructSizesForBuffer.find(StructTy);
  if (SizeIt != StructSizesForBuffer.end())
    return SizeIt->getSecond();

  // if not, calculate the struct layout and create a metadata node
  LLVMContext &Ctx = CGM.getLLVMContext();
  SmallVector<llvm::Metadata *> LayoutItems;

  // start metadata list with a struct name and reserve one slot for its size
  LayoutItems.push_back(MDString::get(Ctx, StructTy->getName()));
  LayoutItems.push_back(nullptr);

  // add element offsets
  size_t StructSize = 0;
  for (llvm::Type *ElTy : StructTy->elements()) {
    size_t Offset = calculateBufferElementOffset(ElTy, &StructSize);
    LayoutItems.push_back(getConstIntMetadata(CGM.getLLVMContext(), Offset));
  }
  // set the size of the buffer to the reserved slot
  LayoutItems[1] = getConstIntMetadata(Ctx, StructSize);

  // add the struct layout to metadata
  CGM.getModule()
      .getOrInsertNamedMetadata("hlsl.cblayouts")
      ->addOperand(MDNode::get(CGM.getLLVMContext(), LayoutItems));

  // add struct size to list and return it
  StructSizesForBuffer[StructTy] = StructSize;
  return StructSize;
}

// Calculates offset of a single element in constant buffer layout.
// The provided LayoutEndOffset marks the end of the layout so far (end offset
// of the buffer or struct). After the element offset calculations are done it
// will be updated the new end of layout value.
// If the PackoffsetAttrs is not nullptr the offset will be based on the
// packoffset annotation.
size_t CGHLSLRuntime::calculateBufferElementOffset(
    llvm::Type *LayoutTy, size_t *LayoutEndOffset,
    HLSLPackOffsetAttr *PackoffsetAttr) {

  size_t ElemOffset = 0;
  size_t ElemSize = 0;
  size_t ArrayCount = 1;
  size_t ArrayStride = 0;
  size_t EndOffset = *LayoutEndOffset;
  size_t NextRowOffset = llvm::alignTo(EndOffset, 16U);

  if (LayoutTy->isArrayTy()) {
    llvm::Type *Ty = LayoutTy;
    while (Ty->isArrayTy()) {
      ArrayCount *= Ty->getArrayNumElements();
      Ty = Ty->getArrayElementType();
    }
    ElemSize =
        Ty->isStructTy()
            ? getOrCalculateStructSizeForBuffer(cast<llvm::StructType>(Ty))
            : getScalarOrVectorSize(Ty);
    ArrayStride = llvm::alignTo(ElemSize, 16U);
    ElemOffset =
        PackoffsetAttr ? PackoffsetAttr->getOffsetInBytes() : NextRowOffset;

  } else if (LayoutTy->isStructTy()) {
    ElemOffset =
        PackoffsetAttr ? PackoffsetAttr->getOffsetInBytes() : NextRowOffset;
    ElemSize =
        getOrCalculateStructSizeForBuffer(cast<llvm::StructType>(LayoutTy));

  } else {
    size_t Align = 0;
    if (LayoutTy->isVectorTy()) {
      llvm::FixedVectorType *FVT = cast<llvm::FixedVectorType>(LayoutTy);
      size_t SubElemSize = FVT->getElementType()->getScalarSizeInBits() / 8;
      ElemSize = FVT->getNumElements() * SubElemSize;
      Align = SubElemSize;
    } else {
      assert(LayoutTy->isIntegerTy() || LayoutTy->isFloatingPointTy());
      ElemSize = LayoutTy->getScalarSizeInBits() / 8;
      Align = ElemSize;
    }
    if (PackoffsetAttr) {
      ElemOffset = PackoffsetAttr->getOffsetInBytes();
    } else {
      ElemOffset = llvm::alignTo(EndOffset, Align);
      // if the element does not fit, move it to the next row
      if (ElemOffset + ElemSize > NextRowOffset)
        ElemOffset = NextRowOffset;
    }
  }

  // Update end offset of the buffer/struct layout; do not update it if
  // the provided EndOffset is already bigger than the new one value
  // (which may happen with packoffset annotations)
  unsigned NewEndOffset =
      ElemOffset + (ArrayCount - 1) * ArrayStride + ElemSize;
  *LayoutEndOffset = std::max<size_t>(EndOffset, NewEndOffset);

  return ElemOffset;
}

// Emits constant global variables for buffer declarations, creates metadata
// linking the constant globals with the buffer. Also calculates the buffer
// layout and creates metadata node describing it.
void CGHLSLRuntime::emitBufferGlobalsAndMetadata(const HLSLBufferDecl *BufDecl,
                                                 llvm::GlobalVariable *BufGV) {
  LLVMContext &Ctx = CGM.getLLVMContext();
  llvm::StructType *LayoutStruct = cast<llvm::StructType>(
      cast<llvm::TargetExtType>(BufGV->getValueType())->getTypeParameter(0));

  // Start metadata list associating the buffer global variable with its
  // constatns
  SmallVector<llvm::Metadata *> BufGlobals;
  BufGlobals.push_back(ValueAsMetadata::get(BufGV));

  // Start layout metadata list with a struct name and reserve one slot for
  // the buffer size
  SmallVector<llvm::Metadata *> LayoutItems;
  LayoutItems.push_back(MDString::get(Ctx, LayoutStruct->getName()));
  LayoutItems.push_back(nullptr);

  size_t BufferSize = 0;
  bool UsePackoffset = BufDecl->hasPackoffset();
  const auto *ElemIt = LayoutStruct->element_begin();
  for (Decl *D : BufDecl->decls()) {
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
          isResourceRecordTypeOrArrayOf(VDTy.getTypePtr())) {
        // Emit static and groupshared variables and resource classes inside
        // cbuffer as regular globals
        CGM.EmitGlobal(VD);
      }
      // Anything else that is not in the hlsl_constant address space must be
      // an empty struct or a zero-sized array and can be ignored
      continue;
    }

    assert(ElemIt != LayoutStruct->element_end() &&
           "number of elements in layout struct does not match");
    llvm::Type *LayoutType = *ElemIt++;

    // Make sure the type of the VarDecl type matches the type of the layout
    // struct element, or that it is a layout struct with the same name
    assert((CGM.getTypes().ConvertTypeForMem(VDTy) == LayoutType ||
            (LayoutType->isStructTy() &&
             cast<llvm::StructType>(LayoutType)
                 ->getName()
                 .starts_with(("struct.__cblayout_" +
                               VDTy->getAsCXXRecordDecl()->getName())
                                  .str()))) &&
           "layout type does not match the converted element type");

    // there might be resources inside the used defined structs
    if (VDTy->isStructureType() && VDTy->isHLSLIntangibleType())
      // FIXME: handle resources in cbuffer structs
      llvm_unreachable("resources in cbuffer are not supported yet");

    // create global variable for the constant and to metadata list
    GlobalVariable *ElemGV =
        cast<GlobalVariable>(CGM.GetAddrOfGlobalVar(VD, LayoutType));
    BufGlobals.push_back(ValueAsMetadata::get(ElemGV));

    // get offset of the global and and to metadata list
    assert(((UsePackoffset && VD->hasAttr<HLSLPackOffsetAttr>()) ||
            !UsePackoffset) &&
           "expected packoffset attribute on every declaration");
    size_t Offset = calculateBufferElementOffset(
        LayoutType, &BufferSize,
        UsePackoffset ? VD->getAttr<HLSLPackOffsetAttr>() : nullptr);
    LayoutItems.push_back(getConstIntMetadata(Ctx, Offset));
  }
  assert(ElemIt == LayoutStruct->element_end() &&
         "number of elements in layout struct does not match");
  // set the size of the buffer
  LayoutItems[1] = getConstIntMetadata(Ctx, BufferSize);

  // add buffer metadata to the module
  CGM.getModule()
      .getOrInsertNamedMetadata("hlsl.cbs")
      ->addOperand(MDNode::get(Ctx, BufGlobals));

  CGM.getModule()
      .getOrInsertNamedMetadata("hlsl.cblayouts")
      ->addOperand(MDNode::get(Ctx, LayoutItems));
}

// Creates resource handle type for the HLSL buffer declaration
static const clang::HLSLAttributedResourceType *
createBufferHandleType(const HLSLBufferDecl *BufDecl) {
  ASTContext &AST = BufDecl->getASTContext();
  QualType QT = AST.getHLSLAttributedResourceType(
      AST.HLSLResourceTy,
      QualType(BufDecl->getLayoutStruct()->getTypeForDecl(), 0),
      HLSLAttributedResourceType::Attributes(ResourceClass::CBuffer));
  return cast<HLSLAttributedResourceType>(QT.getTypePtr());
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
  llvm::Module &M = CGM.getModule();
  llvm::TargetExtType *TargetTy =
      cast<llvm::TargetExtType>(convertHLSLSpecificType(ResHandleTy));
  llvm::GlobalVariable *BufGV =
      new GlobalVariable(TargetTy, /*isConstant*/ true,
                         GlobalValue::LinkageTypes::ExternalLinkage, nullptr,
                         llvm::formatv("{0}{1}", BufDecl->getName(),
                                       BufDecl->isCBuffer() ? ".cb" : ".tb"),
                         GlobalValue::NotThreadLocal);
  M.insertGlobalVariable(BufGV);

  // Add globals for constant buffer elements and create metadata nodes
  emitBufferGlobalsAndMetadata(BufDecl, BufGV);

  // Resource initialization
  const HLSLResourceBindingAttr *RBA =
      BufDecl->getAttr<HLSLResourceBindingAttr>();
  // FIXME: handle implicit binding if no binding attribute is found
  if (RBA)
    createResourceInitFn(CGM, BufGV, RBA->getSlotNumber(),
                         RBA->getSpaceNumber());
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
}

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
    llvm::Function *GroupIDIntrinsic = CGM.getIntrinsic(getGroupIdIntrinsic());
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

static void createResourceInitFn(CodeGenModule &CGM, llvm::GlobalVariable *GV,
                                 unsigned Slot, unsigned Space) {
  LLVMContext &Ctx = CGM.getLLVMContext();
  llvm::Type *Int1Ty = llvm::Type::getInt1Ty(Ctx);

  llvm::Function *InitResFunc = llvm::Function::Create(
      llvm::FunctionType::get(CGM.VoidTy, false),
      llvm::GlobalValue::InternalLinkage,
      ("_init_resource_" + GV->getName()).str(), CGM.getModule());
  InitResFunc->addFnAttr(llvm::Attribute::AlwaysInline);

  llvm::BasicBlock *EntryBB =
      llvm::BasicBlock::Create(Ctx, "entry", InitResFunc);
  CGBuilderTy Builder(CGM, Ctx);
  const DataLayout &DL = CGM.getModule().getDataLayout();
  Builder.SetInsertPoint(EntryBB);

  // Make sure the global variable is resource handle (cbuffer) or
  // resource class (=class where the first element is a resource handle).
  llvm::Type *HandleTy = GV->getValueType();
  assert((HandleTy->isTargetExtTy() ||
          (HandleTy->isStructTy() &&
           HandleTy->getStructElementType(0)->isTargetExtTy())) &&
         "unexpected type of the global");
  if (!HandleTy->isTargetExtTy())
    HandleTy = HandleTy->getStructElementType(0);

  llvm::Value *Args[] = {
      llvm::ConstantInt::get(CGM.IntTy, Space), /* reg_space */
      llvm::ConstantInt::get(CGM.IntTy, Slot),  /* lower_bound */
      // FIXME: resource arrays are not yet implemented
      llvm::ConstantInt::get(CGM.IntTy, 1), /* range_size */
      llvm::ConstantInt::get(CGM.IntTy, 0), /* index */
      // FIXME: NonUniformResourceIndex bit is not yet implemented
      llvm::ConstantInt::get(Int1Ty, false) /* non-uniform */
  };
  llvm::Value *CreateHandle = Builder.CreateIntrinsic(
      /*ReturnType=*/HandleTy,
      CGM.getHLSLRuntime().getCreateHandleFromBindingIntrinsic(), Args, nullptr,
      Twine(GV->getName()).concat("_h"));

  llvm::Value *HandleRef = Builder.CreateStructGEP(GV->getValueType(), GV, 0);
  Builder.CreateAlignedStore(CreateHandle, HandleRef,
                             HandleRef->getPointerAlignment(DL));
  Builder.CreateRetVoid();

  CGM.AddCXXGlobalInit(InitResFunc);
}

void CGHLSLRuntime::handleGlobalVarDefinition(const VarDecl *VD,
                                              llvm::GlobalVariable *GV) {

  // If the global variable has resource binding, create an init function
  // for the resource
  const HLSLResourceBindingAttr *RBA = VD->getAttr<HLSLResourceBindingAttr>();
  if (!RBA)
    // FIXME: collect unbound resources for implicit binding resolution later
    // on?
    return;

  if (!isResourceRecordType(VD->getType().getTypePtr()))
    // FIXME: Only simple declarations of resources are supported for now.
    // Arrays of resources or resources in user defined classes are
    // not implemented yet.
    return;

  createResourceInitFn(CGM, GV, RBA->getSlotNumber(), RBA->getSpaceNumber());
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
