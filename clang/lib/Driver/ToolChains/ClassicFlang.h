//===--- ClassicFlang.h - Flang ToolChain Implementations -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_ClassicFlang_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_ClassicFlang_H

#include "MSVC.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Tool.h"
#include "clang/Driver/Types.h"
#include "llvm/Frontend/Debug/Options.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

namespace clang {
namespace driver {

namespace tools {

/// \brief Flang Fortran frontend
class LLVM_LIBRARY_VISIBILITY ClassicFlang : public Tool {
public:
  ClassicFlang(const ToolChain &TC)
      : Tool("flang:frontend", "Fortran frontend to LLVM", TC) {}

  bool hasGoodDiagnostics() const override { return true; }
  bool hasIntegratedAssembler() const override { return false; }
  bool hasIntegratedCPP() const override { return true; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

} // end namespace tools

} // end namespace driver
} // end namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_ClassicFlang_H
