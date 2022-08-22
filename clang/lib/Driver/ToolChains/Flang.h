//===--- Flang.h - Flang Tool and ToolChain Implementations ====-*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_FLANG_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_FLANG_H

#include "clang/Driver/Tool.h"
#include "clang/Driver/Action.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/ToolChain.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/Compiler.h"

namespace clang {
namespace driver {

namespace tools {

/// Flang compiler tool.
class LLVM_LIBRARY_VISIBILITY Flang : public Tool {
private:
  /// Extract fortran dialect options from the driver arguments and add them to
  /// the list of arguments for the generated command/job.
  ///
  /// \param [in] Args The list of input driver arguments
  /// \param [out] CmdArgs The list of output command arguments
  void AddFortranDialectOptions(const llvm::opt::ArgList &Args,
                                llvm::opt::ArgStringList &CmdArgs) const;

  /// Extract preprocessing options from the driver arguments and add them to
  /// the preprocessor command arguments.
  ///
  /// \param [in] Args The list of input driver arguments
  /// \param [out] CmdArgs The list of output command arguments
  void AddPreprocessingOptions(const llvm::opt::ArgList &Args,
                               llvm::opt::ArgStringList &CmdArgs) const;

  /// Extract PIC options from the driver arguments and add them to
  /// the command arguments.
  ///
  /// \param [in] Args The list of input driver arguments
  /// \param [out] CmdArgs The list of output command arguments
  void AddPicOptions(const llvm::opt::ArgList &Args,
                     llvm::opt::ArgStringList &CmdArgs) const;

  /// Extract other compilation options from the driver arguments and add them
  /// to the command arguments.
  ///
  /// \param [in] Args The list of input driver arguments
  /// \param [out] CmdArgs The list of output command arguments
  void AddOtherOptions(const llvm::opt::ArgList &Args,
                       llvm::opt::ArgStringList &CmdArgs) const;

public:
  Flang(const ToolChain &TC);
  ~Flang() override;

  bool hasGoodDiagnostics() const override { return true; }
  bool hasIntegratedAssembler() const override { return true; }
  bool hasIntegratedCPP() const override { return true; }
  bool canEmitIR() const override { return true; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

} // end namespace tools

} // end namespace driver
} // end namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_FLANG_H
