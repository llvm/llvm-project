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
#include "clang/AST/TypeBase.h"

using namespace clang;
using namespace CodeGen;

SanitizerMetadata::SanitizerMetadata(CodeGenModule &CGM) : CGM(CGM) {}

static bool isAsanHwasanMemTagOrTysan(const SanitizerSet &SS) {
  return SS.hasOneOf(SanitizerKind::Address | SanitizerKind::KernelAddress |
                     SanitizerKind::HWAddress | SanitizerKind::MemTag |
                     SanitizerKind::Type);
}

static SanitizerMask expandKernelSanitizerMasks(SanitizerMask Mask) {
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
  if (!isAsanHwasanMemTagOrTysan(FsanitizeArgument))
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

  Meta.Memtag |=
      static_cast<bool>(FsanitizeArgument.Mask & SanitizerKind::MemtagGlobals);
  Meta.Memtag &= !NoSanitizeAttrSet.hasOneOf(SanitizerKind::MemTag);
  Meta.Memtag &= !CGM.isInNoSanitizeList(
      FsanitizeArgument.Mask & SanitizerKind::MemTag, GV, Loc, Ty);

  Meta.IsDynInit = IsDynInit && !Meta.NoAddress &&
                   FsanitizeArgument.has(SanitizerKind::Address) &&
                   !CGM.isInNoSanitizeList(SanitizerKind::Address |
                                               SanitizerKind::KernelAddress,
                                           GV, Loc, Ty, "init");

  GV->setSanitizerMetadata(Meta);

  if (Ty.isNull() || !CGM.getLangOpts().Sanitize.has(SanitizerKind::Type) ||
      NoSanitizeAttrMask & SanitizerKind::Type)
    return;

  llvm::MDNode *TBAAInfo = CGM.getTBAATypeInfo(Ty);
  if (!TBAAInfo || TBAAInfo == CGM.getTBAATypeInfo(CGM.getContext().CharTy))
    return;

  llvm::Metadata *GlobalMetadata[] = {llvm::ConstantAsMetadata::get(GV),
                                      TBAAInfo};

  // Metadata for the global already registered.
  if (llvm::MDNode::getIfExists(CGM.getLLVMContext(), GlobalMetadata))
    return;

  llvm::MDNode *ThisGlobal =
      llvm::MDNode::get(CGM.getLLVMContext(), GlobalMetadata);
  llvm::NamedMDNode *TysanGlobals =
      CGM.getModule().getOrInsertNamedMetadata("llvm.tysan.globals");
  TysanGlobals->addOperand(ThisGlobal);
}

void SanitizerMetadata::reportGlobal(llvm::GlobalVariable *GV, const VarDecl &D,
                                     bool IsDynInit) {
  if (!isAsanHwasanMemTagOrTysan(CGM.getLangOpts().Sanitize))
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

    // External definitions and incomplete types get handled at the place they
    // are defined.
    if (D.hasExternalStorage() || D.getType()->isIncompleteType())
      NoSanitizeMask |= SanitizerKind::Type;

    return NoSanitizeMask;
  };

  reportGlobal(GV, D.getLocation(), QualName, D.getType(), getNoSanitizeMask(D),
               IsDynInit);
}

void SanitizerMetadata::disableSanitizerForGlobal(llvm::GlobalVariable *GV) {
  reportGlobal(GV, SourceLocation(), "", QualType(), SanitizerKind::All);
}
