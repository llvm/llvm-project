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

using namespace clang;
using namespace CodeGen;

SanitizerMetadata::SanitizerMetadata(CodeGenModule &CGM) : CGM(CGM) {}

static bool isAsanHwasanOrMemTag(const SanitizerSet &SS) {
  return SS.hasOneOf(SanitizerKind::Address | SanitizerKind::KernelAddress |
                     SanitizerKind::HWAddress | SanitizerKind::MemTag);
}

static SanitizerMask expandKernelSanitizerMasks(SanitizerMask Mask) {
  if (Mask & (SanitizerKind::Address | SanitizerKind::KernelAddress))
    Mask |= SanitizerKind::Address | SanitizerKind::KernelAddress;
  // Note: KHWASan doesn't support globals.
  return Mask;
}

static bool shouldTagGlobal(const llvm::GlobalVariable &G) {
  // For now, don't instrument constant data, as it'll be in .rodata anyway. It
  // may be worth instrumenting these in future to stop them from being used as
  // gadgets.
  if (G.getName().starts_with("llvm.") || G.isThreadLocal() || G.isConstant())
    return false;

  // Globals can be placed implicitly or explicitly in sections. There's two
  // different types of globals that meet this criteria that cause problems:
  //  1. Function pointers that are going into various init arrays (either
  //     explicitly through `__attribute__((section(<foo>)))` or implicitly
  //     through `__attribute__((constructor)))`, such as ".(pre)init(_array)",
  //     ".fini(_array)", ".ctors", and ".dtors". These function pointers end up
  //     overaligned and overpadded, making iterating over them problematic, and
  //     each function pointer is individually tagged (so the iteration over
  //     them causes SIGSEGV/MTE[AS]ERR).
  //  2. Global variables put into an explicit section, where the section's name
  //     is a valid C-style identifier. The linker emits a `__start_<name>` and
  //     `__stop_<name>` symbol for the section, so that you can iterate over
  //     globals within this section. Unfortunately, again, these globals would
  //     be tagged and so iteration causes SIGSEGV/MTE[AS]ERR.
  //
  // To mitigate both these cases, and because specifying a section is rare
  // outside of these two cases, disable MTE protection for globals in any
  // section.
  if (G.hasSection())
    return false;

  return true;
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

  if (shouldTagGlobal(*GV)) {
    Meta.Memtag |= static_cast<bool>(FsanitizeArgument.Mask &
                                     SanitizerKind::MemtagGlobals);
    Meta.Memtag &= !NoSanitizeAttrSet.hasOneOf(SanitizerKind::MemTag);
    Meta.Memtag &= !CGM.isInNoSanitizeList(
        FsanitizeArgument.Mask & SanitizerKind::MemTag, GV, Loc, Ty);
  } else {
    Meta.Memtag = false;
  }

  Meta.IsDynInit = IsDynInit && !Meta.NoAddress &&
                   FsanitizeArgument.has(SanitizerKind::Address) &&
                   !CGM.isInNoSanitizeList(SanitizerKind::Address |
                                               SanitizerKind::KernelAddress,
                                           GV, Loc, Ty, "init");

  GV->setSanitizerMetadata(Meta);
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

  reportGlobal(GV, D.getLocation(), QualName, D.getType(), getNoSanitizeMask(D),
               IsDynInit);
}

void SanitizerMetadata::disableSanitizerForGlobal(llvm::GlobalVariable *GV) {
  reportGlobal(GV, SourceLocation(), "", QualType(), SanitizerKind::All);
}
