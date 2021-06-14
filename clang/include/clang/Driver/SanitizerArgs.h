//===--- SanitizerArgs.h - Arguments for sanitizer tools  -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_DRIVER_SANITIZERARGS_H
#define LLVM_CLANG_DRIVER_SANITIZERARGS_H

#include "clang/Basic/Sanitizers.h"
#include "clang/Driver/Types.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizerOptions.h"
#include <string>
#include <vector>

namespace clang {
namespace driver {

class ToolChain;

class SanitizerArgs {
  SanitizerSet Sanitizers;
  SanitizerSet RecoverableSanitizers;
  SanitizerSet TrapSanitizers;

  std::vector<std::string> UserIgnorelistFiles;
  std::vector<std::string> SystemIgnorelistFiles;
  std::vector<std::string> CoverageAllowlistFiles;
  std::vector<std::string> CoverageIgnorelistFiles;
  int CoverageFeatures = 0;
  int MsanTrackOrigins = 0;
  bool MsanUseAfterDtor = true;
  bool CfiCrossDso = false;
  bool CfiICallGeneralizePointers = false;
  bool CfiCanonicalJumpTables = false;
  int AsanFieldPadding = 0;
  bool SharedRuntime = false;
  bool AsanUseAfterScope = true;
  bool AsanPoisonCustomArrayCookie = false;
  bool AsanGlobalsDeadStripping = false;
  bool AsanUseOdrIndicator = false;
  bool AsanInvalidPointerCmp = false;
  bool AsanInvalidPointerSub = false;
  llvm::AsanDtorKind AsanDtorKind = llvm::AsanDtorKind::Invalid;
  std::string HwasanAbi;
  bool LinkRuntimes = true;
  bool LinkCXXRuntimes = false;
  bool NeedPIE = false;
  bool SafeStackRuntime = false;
  bool Stats = false;
  bool TsanMemoryAccess = true;
  bool TsanFuncEntryExit = true;
  bool TsanAtomics = true;
  bool MinimalRuntime = false;
  // True if cross-dso CFI support if provided by the system (i.e. Android).
  bool ImplicitCfiRuntime = false;
  bool NeedsMemProfRt = false;
  bool HwasanUseAliases = false;
  llvm::AsanDetectStackUseAfterReturnMode AsanUseAfterReturn =
      llvm::AsanDetectStackUseAfterReturnMode::Invalid;

public:
  /// Parses the sanitizer arguments from an argument list.
  SanitizerArgs(const ToolChain &TC, const llvm::opt::ArgList &Args);

  bool needsSharedRt() const { return SharedRuntime; }

  bool needsMemProfRt() const { return NeedsMemProfRt; }
  bool needsAsanRt() const { return Sanitizers.has(SanitizerKind::Address); }
  bool needsHwasanRt() const {
    return Sanitizers.has(SanitizerKind::HWAddress);
  }
  bool needsHwasanAliasesRt() const {
    return needsHwasanRt() && HwasanUseAliases;
  }
  bool needsTsanRt() const { return Sanitizers.has(SanitizerKind::Thread); }
  bool needsMsanRt() const { return Sanitizers.has(SanitizerKind::Memory); }
  bool needsFuzzer() const { return Sanitizers.has(SanitizerKind::Fuzzer); }
  bool needsLsanRt() const {
    return Sanitizers.has(SanitizerKind::Leak) &&
           !Sanitizers.has(SanitizerKind::Address) &&
           !Sanitizers.has(SanitizerKind::HWAddress);
  }
  bool needsFuzzerInterceptors() const;
  bool needsUbsanRt() const;
  bool requiresMinimalRuntime() const { return MinimalRuntime; }
  bool needsDfsanRt() const { return Sanitizers.has(SanitizerKind::DataFlow); }
  bool needsSafeStackRt() const { return SafeStackRuntime; }
  bool needsCfiRt() const;
  bool needsCfiDiagRt() const;
  bool needsStatsRt() const { return Stats; }
  bool needsScudoRt() const { return Sanitizers.has(SanitizerKind::Scudo); }

  bool requiresPIE() const;
  bool needsUnwindTables() const;
  bool needsLTO() const;
  bool linkRuntimes() const { return LinkRuntimes; }
  bool linkCXXRuntimes() const { return LinkCXXRuntimes; }
  bool hasCrossDsoCfi() const { return CfiCrossDso; }
  bool hasAnySanitizer() const { return !Sanitizers.empty(); }
  void addArgs(const ToolChain &TC, const llvm::opt::ArgList &Args,
               llvm::opt::ArgStringList &CmdArgs, types::ID InputType) const;
};

}  // namespace driver
}  // namespace clang

#endif
