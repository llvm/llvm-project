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
#include "CodeGenModule.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/TargetOptions.h"
#include "llvm/IR/IntrinsicsDirectX.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"

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
} // namespace

void CGHLSLRuntime::finishCodeGen() {
  auto &TargetOpts = CGM.getTarget().getTargetOpts();
  llvm::Module &M = CGM.getModule();
  Triple T(M.getTargetTriple());
  if (T.getArch() == Triple::ArchType::dxil)
    addDxilValVersion(TargetOpts.DxilValidatorVersion, M);
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
  ResourceMD->addOperand(MDNode::get(
      Ctx, {ValueAsMetadata::get(GV), MDString::get(Ctx, QT.getAsString()),
            ConstantAsMetadata::get(B.getInt32(Counter))}));
}

void clang::CodeGen::CGHLSLRuntime::setHLSLEntryAttributes(
    const FunctionDecl *FD, llvm::Function *Fn) {
  const auto *ShaderAttr = FD->getAttr<HLSLShaderAttr>();
  assert(ShaderAttr && "All entry functions must have a HLSLShaderAttr");
  const StringRef ShaderAttrKindStr = "hlsl.shader";
  Fn->addFnAttr(ShaderAttrKindStr,
                ShaderAttr->ConvertShaderTypeToStr(ShaderAttr->getType()));
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
  for (const auto *Param : FD->parameters()) {
    Args.push_back(emitInputSemantic(B, *Param));
  }

  CallInst *CI = B.CreateCall(FunctionCallee(Fn), Args);
  (void)CI;
  // FIXME: Handle codegen for return type semantics.
  B.CreateRetVoid();
}
