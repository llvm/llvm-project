//===--- SanitizerArgs.h - Arguments for sanitizer tools  -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_DRIVER_SANITIZERARGS_H
#define LLVM_CLANG_DRIVER_SANITIZERARGS_H

#include "clang/Basic/Sanitizers.h"
#include "clang/Driver/Types.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include <string>
#include <vector>

namespace clang {
namespace driver {

class ToolChain;

class SanitizerArgs {
  SanitizerSet Sanitizers;
  SanitizerSet RecoverableSanitizers;
  SanitizerSet TrapSanitizers;

  std::vector<std::string> BlacklistFiles;
  std::vector<std::string> ExtraDeps;
  int CoverageFeatures = 0;
  int MsanTrackOrigins = 0;
  bool MsanUseAfterDtor = false;
  bool CfiCrossDso = false;
  int AsanFieldPadding = 0;
  bool AsanSharedRuntime = false;
  bool LinkCXXRuntimes = false;
  bool NeedPIE = false;

 public:
  /// Parses the sanitizer arguments from an argument list.
  SanitizerArgs(const ToolChain &TC, const llvm::opt::ArgList &Args);

  bool needsAsanRt() const { return Sanitizers.has(SanitizerKind::Address); }
  bool needsSharedAsanRt() const { return AsanSharedRuntime; }
  bool needsTsanRt() const { return Sanitizers.has(SanitizerKind::Thread); }
  bool needsMsanRt() const { return Sanitizers.has(SanitizerKind::Memory); }
  bool needsLsanRt() const {
    return Sanitizers.has(SanitizerKind::Leak) &&
           !Sanitizers.has(SanitizerKind::Address);
  }
  bool needsUbsanRt() const;
  bool needsDfsanRt() const { return Sanitizers.has(SanitizerKind::DataFlow); }
  bool needsSafeStackRt() const {
    return Sanitizers.has(SanitizerKind::SafeStack);
  }
  bool needsCfiRt() const;
  bool needsCfiDiagRt() const;

  bool requiresPIE() const;
  bool needsUnwindTables() const;
  bool linkCXXRuntimes() const { return LinkCXXRuntimes; }
  void addArgs(const ToolChain &TC, const llvm::opt::ArgList &Args,
               llvm::opt::ArgStringList &CmdArgs, types::ID InputType) const;
};

}  // namespace driver
}  // namespace clang

#endif
