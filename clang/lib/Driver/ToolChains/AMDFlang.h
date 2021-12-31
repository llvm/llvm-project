//===--- AMDFlang.h - Flang Tool and ToolChain Implementations =-*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_AMDFLANG_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_AMDFLANG_H

#include "clang/Driver/Tool.h"
#include "clang/Driver/Action.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/ToolChain.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/Compiler.h"

namespace clang {
class ObjCRuntime;

namespace driver {

namespace tools {

/// Flang compiler tool.
class LLVM_LIBRARY_VISIBILITY AMDFlang : public Tool {
public:
  AMDFlang(const ToolChain &TC);
  ~AMDFlang() override;

  bool hasGoodDiagnostics() const override { return true; }
  bool hasIntegratedAssembler() const override { return true; }
  bool hasIntegratedCPP() const override { return true; }
  bool canEmitIR() const override { return true; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
private:
  void addWaveSizeToFlangArgs(const llvm::opt::ArgList &DriverArgs,
                         llvm::opt::ArgStringList &FlangArgs) const;

  void addTargetArchToFlangArgs(const llvm::opt::ArgList &DriverArgs,
                         llvm::opt::ArgStringList &FlangArgs) const;
};

} // end namespace tools

} // end namespace driver
} // end namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_AMDFLANG_H
