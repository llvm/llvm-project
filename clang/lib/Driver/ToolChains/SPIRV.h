//===--- SPIRV.h - SPIR-V Tool Implementations ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_SPIRV_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_SPIRV_H

#include "clang/Driver/Tool.h"
#include "clang/Driver/ToolChain.h"

namespace clang {
namespace driver {
namespace tools {
namespace SPIRV {

void addTranslatorArgs(const llvm::opt::ArgList &InArgs,
                       llvm::opt::ArgStringList &OutArgs);

void constructTranslateCommand(Compilation &C, const Tool &T,
                               const JobAction &JA, const InputInfo &Output,
                               const InputInfo &Input,
                               const llvm::opt::ArgStringList &Args);

class LLVM_LIBRARY_VISIBILITY Translator : public Tool {
public:
  Translator(const ToolChain &TC)
      : Tool("SPIR-V::Translator", "llvm-spirv", TC) {}

  bool hasIntegratedCPP() const override { return false; }
  bool hasIntegratedAssembler() const override { return true; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

} // namespace SPIRV
} // namespace tools
} // namespace driver
} // namespace clang
#endif
