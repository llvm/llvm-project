//===--- SanitizerMetadata.cpp - Ignored entities for sanitizers ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Class which emits metadata consumed by sanitizer instrumentation passes.
//
//===----------------------------------------------------------------------===//
#include "SanitizerMetadata.h"
#include "CodeGenModule.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Type.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"

using namespace clang;
using namespace CodeGen;

SanitizerMetadata::SanitizerMetadata(CodeGenModule &CGM) : CGM(CGM) {}

// TODO(hctim): Can be removed when we migrate off of llvm.asan.globals. This
// prevents llvm.asan.globals from being emitted for
// __attribute__((disable_sanitizer_instrumentation)) and uses of
// -fsanitize-ignorelist when a sanitizer isn't enabled.
static bool isAsanHwasanOrMemTag(const SanitizerSet &SS) {
  return SS.hasOneOf(SanitizerKind::Address | SanitizerKind::KernelAddress |
                     SanitizerKind::HWAddress | SanitizerKind::MemTag);
}

SanitizerMask expandKernelSanitizerMasks(SanitizerMask Mask) {
  if (Mask & (SanitizerKind::Address | SanitizerKind::KernelAddress))
    Mask |= SanitizerKind::Address | SanitizerKind::KernelAddress;
  // Note: KHWASan doesn't support globals.
  return Mask;
}

void SanitizerMetadata::reportGlobal(llvm::GlobalVariable *GV,
                                     SourceLocation Loc, StringRef Name,
                                     QualType Ty,
                                     SanitizerMask NoSanitizeAttrMask,
                                     bool IsDynInit) {
  SanitizerSet FsanitizeArgument = CGM.getLangOpts().Sanitize;
  if (!isAsanHwasanOrMemTag(FsanitizeArgument))
    return;

  FsanitizeArgument.Mask = expandKernelSanitizerMasks(FsanitizeArgument.Mask);
  NoSanitizeAttrMask = expandKernelSanitizerMasks(NoSanitizeAttrMask);
  SanitizerSet NoSanitizeAttrSet = {NoSanitizeAttrMask &
                                    FsanitizeArgument.Mask};

  llvm::GlobalVariable::SanitizerMetadata Meta;
  if (GV->hasSanitizerMetadata())
    Meta = GV->getSanitizerMetadata();

  Meta.NoAddress |= NoSanitizeAttrSet.hasOneOf(SanitizerKind::Address);
  Meta.NoAddress |= CGM.isInNoSanitizeList(
      FsanitizeArgument.Mask & SanitizerKind::Address, GV, Loc, Ty);

  Meta.NoHWAddress |= NoSanitizeAttrSet.hasOneOf(SanitizerKind::HWAddress);
  Meta.NoHWAddress |= CGM.isInNoSanitizeList(
      FsanitizeArgument.Mask & SanitizerKind::HWAddress, GV, Loc, Ty);

  Meta.NoMemtag |= NoSanitizeAttrSet.hasOneOf(SanitizerKind::MemTag);
  Meta.NoMemtag |= CGM.isInNoSanitizeList(
      FsanitizeArgument.Mask & SanitizerKind::MemTag, GV, Loc, Ty);

  if (FsanitizeArgument.has(SanitizerKind::Address)) {
    // TODO(hctim): Make this conditional when we migrate off llvm.asan.globals.
    IsDynInit &= !CGM.isInNoSanitizeList(SanitizerKind::Address |
                                             SanitizerKind::KernelAddress,
                                         GV, Loc, Ty, "init");
    Meta.IsDynInit = IsDynInit;
  }

  bool IsExcluded = Meta.NoAddress || Meta.NoHWAddress || Meta.NoMemtag;

  GV->setSanitizerMetadata(Meta);

  // TODO(hctim): Code below can be removed when we migrate off of
  // llvm.asan.globals onto the new metadata attributes.
  llvm::Metadata *LocDescr = nullptr;
  llvm::Metadata *GlobalName = nullptr;
  llvm::LLVMContext &VMContext = CGM.getLLVMContext();
  if (!IsExcluded) {
    // Don't generate source location and global name if it is on
    // the NoSanitizeList - it won't be instrumented anyway.
    LocDescr = getLocationMetadata(Loc);
    if (!Name.empty())
      GlobalName = llvm::MDString::get(VMContext, Name);
  }

  llvm::Metadata *GlobalMetadata[] = {
      llvm::ConstantAsMetadata::get(GV), LocDescr, GlobalName,
      llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(llvm::Type::getInt1Ty(VMContext), IsDynInit)),
      llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
          llvm::Type::getInt1Ty(VMContext), IsExcluded))};

  llvm::MDNode *ThisGlobal = llvm::MDNode::get(VMContext, GlobalMetadata);
  llvm::NamedMDNode *AsanGlobals =
      CGM.getModule().getOrInsertNamedMetadata("llvm.asan.globals");
  AsanGlobals->addOperand(ThisGlobal);
}

void SanitizerMetadata::reportGlobal(llvm::GlobalVariable *GV, const VarDecl &D,
                                     bool IsDynInit) {
  if (!isAsanHwasanOrMemTag(CGM.getLangOpts().Sanitize))
    return;
  std::string QualName;
  llvm::raw_string_ostream OS(QualName);
  D.printQualifiedName(OS);

  auto getNoSanitizeMask = [](const VarDecl &D) {
    if (D.hasAttr<DisableSanitizerInstrumentationAttr>())
      return SanitizerKind::All;

    SanitizerMask NoSanitizeMask;
    for (auto *Attr : D.specific_attrs<NoSanitizeAttr>())
      NoSanitizeMask |= Attr->getMask();

    return NoSanitizeMask;
  };

  reportGlobal(GV, D.getLocation(), OS.str(), D.getType(), getNoSanitizeMask(D),
               IsDynInit);
}

void SanitizerMetadata::disableSanitizerForGlobal(llvm::GlobalVariable *GV) {
  reportGlobal(GV, SourceLocation(), "", QualType(), SanitizerKind::All);
}

void SanitizerMetadata::disableSanitizerForInstruction(llvm::Instruction *I) {
  I->setMetadata(llvm::LLVMContext::MD_nosanitize,
                 llvm::MDNode::get(CGM.getLLVMContext(), None));
}

llvm::MDNode *SanitizerMetadata::getLocationMetadata(SourceLocation Loc) {
  PresumedLoc PLoc = CGM.getContext().getSourceManager().getPresumedLoc(Loc);
  if (!PLoc.isValid())
    return nullptr;
  llvm::LLVMContext &VMContext = CGM.getLLVMContext();
  llvm::Metadata *LocMetadata[] = {
      llvm::MDString::get(VMContext, PLoc.getFilename()),
      llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
          llvm::Type::getInt32Ty(VMContext), PLoc.getLine())),
      llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
          llvm::Type::getInt32Ty(VMContext), PLoc.getColumn())),
  };
  return llvm::MDNode::get(VMContext, LocMetadata);
}
