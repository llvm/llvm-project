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
#include "clang/AST/Decl.h"
#include "clang/Basic/TargetOptions.h"
#include "llvm/Frontend/HLSL/HLSLResource.h"
#include "llvm/IR/IntrinsicsDirectX.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FormatVariadic.h"

using namespace clang;
using namespace CodeGen;
using namespace hlsl;
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
  GlobalVariable *CBGV =
      new GlobalVariable(Buf.LayoutStruct, /*isConstant*/ true,
                         GlobalValue::LinkageTypes::ExternalLinkage, nullptr,
                         Buf.Name + (Buf.IsCBuffer ? ".cb." : ".tb."),
                         GlobalValue::NotThreadLocal);

  IRBuilder<> B(CBGV->getContext());
  Value *ZeroIdx = B.getInt32(0);
  // Replace Const use with CB use.
  for (auto &[GV, Offset]: Buf.Constants) {
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

} // namespace

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
    M.getGlobalList().push_back(GV);
    // FIXME: generate resource binding.
    // See https://github.com/llvm/llvm-project/issues/57915.
  }
}

CGHLSLRuntime::Buffer::Buffer(const HLSLBufferDecl *D) {
  Name = D->getName();
  IsCBuffer = D->isCBuffer();
  if (auto *Binding = D->getAttr<HLSLResourceBindingAttr>()) {
    llvm::APInt RegInt(64, 0);
    Binding->getSlot().substr(1).getAsInteger(10, RegInt);
    Reg = RegInt.getLimitedValue();

    llvm::APInt SpaceInt(64, 0);
    Binding->getSpace().substr(5).getAsInteger(10, RegInt);
    Space = SpaceInt.getLimitedValue();
  } else {
    Space = 0;
  }
}

void CGHLSLRuntime::annotateHLSLResource(const VarDecl *D, GlobalVariable *GV) {
  const Type *Ty = D->getType()->getPointeeOrArrayElementType();
  if (!Ty)
    return;
  const auto *RD = Ty->getAsCXXRecordDecl();
  if (!RD)
    return;
  const auto *Attr = RD->getAttr<HLSLResourceAttr>();
  if (!Attr)
    return;

  HLSLResourceAttr::ResourceClass RC = Attr->getResourceType();
  uint32_t Counter = ResourceCounters[static_cast<uint32_t>(RC)]++;

  NamedMDNode *ResourceMD = nullptr;
  switch (RC) {
  case HLSLResourceAttr::ResourceClass::UAV:
    ResourceMD = CGM.getModule().getOrInsertNamedMetadata("hlsl.uavs");
    break;
  default:
    assert(false && "Unsupported buffer type!");
    return;
  }

  assert(ResourceMD != nullptr &&
         "ResourceMD must have been set by the switch above.");

  auto &Ctx = CGM.getModule().getContext();
  IRBuilder<> B(Ctx);
  QualType QT(Ty, 0);
  llvm::hlsl::FrontendResource Res(GV, QT.getAsString(), Counter);
  ResourceMD->addOperand(Res.getMetadata());
}

void clang::CodeGen::CGHLSLRuntime::setHLSLEntryAttributes(
    const FunctionDecl *FD, llvm::Function *Fn) {
  const auto *ShaderAttr = FD->getAttr<HLSLShaderAttr>();
  assert(ShaderAttr && "All entry functions must have a HLSLShaderAttr");
  const StringRef ShaderAttrKindStr = "hlsl.shader";
  Fn->addFnAttr(ShaderAttrKindStr,
                ShaderAttr->ConvertShaderTypeToStr(ShaderAttr->getType()));
  if (HLSLNumThreadsAttr *NumThreadsAttr = FD->getAttr<HLSLNumThreadsAttr>()) {
    const StringRef NumThreadsKindStr = "hlsl.numthreads";
    std::string NumThreadsStr =
        formatv("{0},{1},{2}", NumThreadsAttr->getX(), NumThreadsAttr->getY(),
                NumThreadsAttr->getZ());
    Fn->addFnAttr(NumThreadsKindStr, NumThreadsStr);
  }
}

llvm::Value *CGHLSLRuntime::emitInputSemantic(IRBuilder<> &B,
                                              const ParmVarDecl &D) {
  assert(D.hasAttrs() && "Entry parameter missing annotation attribute!");
  if (D.hasAttr<HLSLSV_GroupIndexAttr>()) {
    llvm::Function *DxGroupIndex =
        CGM.getIntrinsic(Intrinsic::dx_flattened_thread_id_in_group);
    return B.CreateCall(FunctionCallee(DxGroupIndex));
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
  // FIXME: support struct parameters where semantics are on members.
  // See: https://github.com/llvm/llvm-project/issues/57874
  for (const auto *Param : FD->parameters()) {
    Args.push_back(emitInputSemantic(B, *Param));
  }

  CallInst *CI = B.CreateCall(FunctionCallee(Fn), Args);
  (void)CI;
  // FIXME: Handle codegen for return type semantics.
  // See: https://github.com/llvm/llvm-project/issues/57875
  B.CreateRetVoid();
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
    IRBuilder<> B(&F.getEntryBlock(), F.getEntryBlock().begin());
    for (auto *Fn : CtorFns)
      B.CreateCall(FunctionCallee(Fn));

    // Insert global dtors before the terminator of the last instruction
    B.SetInsertPoint(F.back().getTerminator());
    for (auto *Fn : DtorFns)
      B.CreateCall(FunctionCallee(Fn));
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
