//===---- TargetInfo.cpp - Encapsulate target details -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These classes wrap the information about a call or function
// definition used to handle ABI compliancy.
//
//===----------------------------------------------------------------------===//

#include "TargetInfo.h"
#include "ABIInfo.h"
#include "ABIInfoImpl.h"
#include "CodeGenFunction.h"
#include "clang/Basic/CodeGenOptions.h"
#include "clang/CodeGen/CGFunctionInfo.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace CodeGen;

LLVM_DUMP_METHOD void ABIArgInfo::dump() const {
  raw_ostream &OS = llvm::errs();
  OS << "(ABIArgInfo Kind=";
  switch (TheKind) {
  case Direct:
    OS << "Direct Type=";
    if (llvm::Type *Ty = getCoerceToType())
      Ty->print(OS);
    else
      OS << "null";
    break;
  case Extend:
    OS << "Extend";
    break;
  case Ignore:
    OS << "Ignore";
    break;
  case InAlloca:
    OS << "InAlloca Offset=" << getInAllocaFieldIndex();
    break;
  case Indirect:
    OS << "Indirect Align=" << getIndirectAlign().getQuantity()
       << " ByVal=" << getIndirectByVal()
       << " Realign=" << getIndirectRealign();
    break;
  case IndirectAliased:
    OS << "Indirect Align=" << getIndirectAlign().getQuantity()
       << " AadrSpace=" << getIndirectAddrSpace()
       << " Realign=" << getIndirectRealign();
    break;
  case Expand:
    OS << "Expand";
    break;
  case CoerceAndExpand:
    OS << "CoerceAndExpand Type=";
    getCoerceAndExpandType()->print(OS);
    break;
  case TargetSpecific:
    OS << "TargetSpecific Type=";
    if (llvm::Type *Ty = getCoerceToType())
      Ty->print(OS);
    else
      OS << "null";
    break;
  }
  OS << ")\n";
}

TargetCodeGenInfo::TargetCodeGenInfo(std::unique_ptr<ABIInfo> Info)
    : Info(std::move(Info)) {}

TargetCodeGenInfo::~TargetCodeGenInfo() = default;

// If someone can figure out a general rule for this, that would be great.
// It's probably just doomed to be platform-dependent, though.
unsigned TargetCodeGenInfo::getSizeOfUnwindException() const {
  if (getABIInfo().getCodeGenOpts().hasSEHExceptions())
    return getABIInfo().getDataLayout().getPointerSizeInBits() > 32 ? 64 : 48;
  // Verified for:
  //   x86-64     FreeBSD, Linux, Darwin
  //   x86-32     FreeBSD, Linux, Darwin
  //   PowerPC    Linux
  //   ARM        Darwin (*not* EABI)
  //   AArch64    Linux
  return 32;
}

bool TargetCodeGenInfo::isNoProtoCallVariadic(const CallArgList &args,
                                     const FunctionNoProtoType *fnType) const {
  // The following conventions are known to require this to be false:
  //   x86_stdcall
  //   MIPS
  // For everything else, we just prefer false unless we opt out.
  return false;
}

void
TargetCodeGenInfo::getDependentLibraryOption(llvm::StringRef Lib,
                                             llvm::SmallString<24> &Opt) const {
  // This assumes the user is passing a library name like "rt" instead of a
  // filename like "librt.a/so", and that they don't care whether it's static or
  // dynamic.
  Opt = "-l";
  Opt += Lib;
}

unsigned TargetCodeGenInfo::getDeviceKernelCallingConv() const {
  if (getABIInfo().getContext().getLangOpts().OpenCL) {
    // Device kernels are called via an explicit runtime API with arguments,
    // such as set with clSetKernelArg() for OpenCL, not as normal
    // sub-functions. Return SPIR_KERNEL by default as the kernel calling
    // convention to ensure the fingerprint is fixed such way that each kernel
    // argument gets one matching argument in the produced kernel function
    // argument list to enable feasible implementation of clSetKernelArg() with
    // aggregates etc. In case we would use the default C calling conv here,
    // clSetKernelArg() might break depending on the target-specific
    // conventions; different targets might split structs passed as values
    // to multiple function arguments etc.
    return llvm::CallingConv::SPIR_KERNEL;
  }
  llvm_unreachable("Unknown kernel calling convention");
}

void TargetCodeGenInfo::setOCLKernelStubCallingConvention(
    const FunctionType *&FT) const {
  FT = getABIInfo().getContext().adjustFunctionType(
      FT, FT->getExtInfo().withCallingConv(CC_C));
}

llvm::Constant *TargetCodeGenInfo::getNullPointer(const CodeGen::CodeGenModule &CGM,
    llvm::PointerType *T, QualType QT) const {
  return llvm::ConstantPointerNull::get(T);
}

LangAS TargetCodeGenInfo::getGlobalVarAddressSpace(CodeGenModule &CGM,
                                                   const VarDecl *D) const {
  assert(!CGM.getLangOpts().OpenCL &&
         !(CGM.getLangOpts().CUDA && CGM.getLangOpts().CUDAIsDevice) &&
         "Address space agnostic languages only");
  return D ? D->getType().getAddressSpace() : LangAS::Default;
}

StringRef
TargetCodeGenInfo::getLLVMSyncScopeStr(const LangOptions &LangOpts,
                                       SyncScope Scope,
                                       llvm::AtomicOrdering Ordering) const {
  return ""; /* default sync scope */
}

llvm::SyncScope::ID
TargetCodeGenInfo::getLLVMSyncScopeID(const LangOptions &LangOpts,
                                      SyncScope Scope,
                                      llvm::AtomicOrdering Ordering,
                                      llvm::LLVMContext &Ctx) const {
  return Ctx.getOrInsertSyncScopeID(
      getLLVMSyncScopeStr(LangOpts, Scope, Ordering));
}

void TargetCodeGenInfo::addStackProbeTargetAttributes(
    const Decl *D, llvm::GlobalValue *GV, CodeGen::CodeGenModule &CGM) const {
  if (llvm::Function *Fn = dyn_cast_or_null<llvm::Function>(GV)) {
    if (CGM.getCodeGenOpts().StackProbeSize != 4096)
      Fn->addFnAttr("stack-probe-size",
                    llvm::utostr(CGM.getCodeGenOpts().StackProbeSize));
    if (CGM.getCodeGenOpts().NoStackArgProbe)
      Fn->addFnAttr("no-stack-arg-probe");
  }
}

/// Create an OpenCL kernel for an enqueued block.
///
/// The kernel has the same function type as the block invoke function. Its
/// name is the name of the block invoke function postfixed with "_kernel".
/// It simply calls the block invoke function then returns.
llvm::Value *TargetCodeGenInfo::createEnqueuedBlockKernel(
    CodeGenFunction &CGF, llvm::Function *Invoke, llvm::Type *BlockTy) const {
  auto *InvokeFT = Invoke->getFunctionType();
  auto &C = CGF.getLLVMContext();
  std::string Name = Invoke->getName().str() + "_kernel";
  auto *FT = llvm::FunctionType::get(llvm::Type::getVoidTy(C),
                                     InvokeFT->params(), false);
  auto *F = llvm::Function::Create(FT, llvm::GlobalValue::ExternalLinkage, Name,
                                   &CGF.CGM.getModule());
  llvm::CallingConv::ID KernelCC =
      CGF.getTypes().ClangCallConvToLLVMCallConv(CallingConv::CC_DeviceKernel);
  F->setCallingConv(KernelCC);

  llvm::AttrBuilder KernelAttrs(C);

  // FIXME: This is missing setTargetAttributes
  CGF.CGM.addDefaultFunctionDefinitionAttributes(KernelAttrs);
  F->addFnAttrs(KernelAttrs);

  auto IP = CGF.Builder.saveIP();
  auto *BB = llvm::BasicBlock::Create(C, "entry", F);
  auto &Builder = CGF.Builder;
  Builder.SetInsertPoint(BB);
  llvm::SmallVector<llvm::Value *, 2> Args(llvm::make_pointer_range(F->args()));
  llvm::CallInst *Call = Builder.CreateCall(Invoke, Args);
  Call->setCallingConv(Invoke->getCallingConv());

  Builder.CreateRetVoid();
  Builder.restoreIP(IP);
  return F;
}

void TargetCodeGenInfo::setBranchProtectionFnAttributes(
    const TargetInfo::BranchProtectionInfo &BPI, llvm::Function &F) {
  // Called on already created and initialized function where attributes already
  // set from command line attributes but some might need to be removed as the
  // actual BPI is different.
  if (BPI.SignReturnAddr != LangOptions::SignReturnAddressScopeKind::None) {
    F.addFnAttr("sign-return-address", BPI.getSignReturnAddrStr());
    F.addFnAttr("sign-return-address-key", BPI.getSignKeyStr());
  } else {
    if (F.hasFnAttribute("sign-return-address"))
      F.removeFnAttr("sign-return-address");
    if (F.hasFnAttribute("sign-return-address-key"))
      F.removeFnAttr("sign-return-address-key");
  }

  auto AddRemoveAttributeAsSet = [&](bool Set, const StringRef &ModAttr) {
    if (Set)
      F.addFnAttr(ModAttr);
    else if (F.hasFnAttribute(ModAttr))
      F.removeFnAttr(ModAttr);
  };

  AddRemoveAttributeAsSet(BPI.BranchTargetEnforcement,
                          "branch-target-enforcement");
  AddRemoveAttributeAsSet(BPI.BranchProtectionPAuthLR,
                          "branch-protection-pauth-lr");
  AddRemoveAttributeAsSet(BPI.GuardedControlStack, "guarded-control-stack");
}

void TargetCodeGenInfo::initBranchProtectionFnAttributes(
    const TargetInfo::BranchProtectionInfo &BPI, llvm::AttrBuilder &FuncAttrs) {
  // Only used for initializing attributes in the AttrBuilder, which will not
  // contain any of these attributes so no need to remove anything.
  if (BPI.SignReturnAddr != LangOptions::SignReturnAddressScopeKind::None) {
    FuncAttrs.addAttribute("sign-return-address", BPI.getSignReturnAddrStr());
    FuncAttrs.addAttribute("sign-return-address-key", BPI.getSignKeyStr());
  }
  if (BPI.BranchTargetEnforcement)
    FuncAttrs.addAttribute("branch-target-enforcement");
  if (BPI.BranchProtectionPAuthLR)
    FuncAttrs.addAttribute("branch-protection-pauth-lr");
  if (BPI.GuardedControlStack)
    FuncAttrs.addAttribute("guarded-control-stack");
}

void TargetCodeGenInfo::setPointerAuthFnAttributes(
    const PointerAuthOptions &Opts, llvm::Function &F) {
  auto UpdateAttr = [&F](bool AttrShouldExist, StringRef AttrName) {
    if (AttrShouldExist && !F.hasFnAttribute(AttrName))
      F.addFnAttr(AttrName);
    if (!AttrShouldExist && F.hasFnAttribute(AttrName))
      F.removeFnAttr(AttrName);
  };
  UpdateAttr(Opts.ReturnAddresses, "ptrauth-returns");
  UpdateAttr((bool)Opts.FunctionPointers, "ptrauth-calls");
  UpdateAttr(Opts.AuthTraps, "ptrauth-auth-traps");
  UpdateAttr(Opts.IndirectGotos, "ptrauth-indirect-gotos");
  UpdateAttr(Opts.AArch64JumpTableHardening, "aarch64-jump-table-hardening");
}

void TargetCodeGenInfo::initPointerAuthFnAttributes(
    const PointerAuthOptions &Opts, llvm::AttrBuilder &FuncAttrs) {
  if (Opts.ReturnAddresses)
    FuncAttrs.addAttribute("ptrauth-returns");
  if (Opts.FunctionPointers)
    FuncAttrs.addAttribute("ptrauth-calls");
  if (Opts.AuthTraps)
    FuncAttrs.addAttribute("ptrauth-auth-traps");
  if (Opts.IndirectGotos)
    FuncAttrs.addAttribute("ptrauth-indirect-gotos");
  if (Opts.AArch64JumpTableHardening)
    FuncAttrs.addAttribute("aarch64-jump-table-hardening");
}

namespace {
class DefaultTargetCodeGenInfo : public TargetCodeGenInfo {
public:
  DefaultTargetCodeGenInfo(CodeGen::CodeGenTypes &CGT)
      : TargetCodeGenInfo(std::make_unique<DefaultABIInfo>(CGT)) {}
};
} // namespace

std::unique_ptr<TargetCodeGenInfo>
CodeGen::createDefaultTargetCodeGenInfo(CodeGenModule &CGM) {
  return std::make_unique<DefaultTargetCodeGenInfo>(CGM.getTypes());
}
