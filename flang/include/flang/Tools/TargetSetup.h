//===-- Tools/TargetSetup.h ------------------------------------- *-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_TOOLS_TARGET_SETUP_H
#define FORTRAN_TOOLS_TARGET_SETUP_H

#include "flang/Evaluate/target.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include <memory>

namespace Fortran::tools {

[[maybe_unused]] inline static void setUpTargetCharacteristics(
    Fortran::evaluate::TargetCharacteristics &targetCharacteristics,
    const llvm::TargetMachine &targetMachine,
    const std::string &compilerVersion, const std::string &compilerOptions) {

  const llvm::Triple &targetTriple{targetMachine.getTargetTriple()};
  // FIXME: Handle real(3) ?
  if (targetTriple.getArch() != llvm::Triple::ArchType::x86_64)
    targetCharacteristics.DisableType(
        Fortran::common::TypeCategory::Real, /*kind=*/10);

  targetCharacteristics.set_compilerOptionsString(compilerOptions)
      .set_compilerVersionString(compilerVersion);

  if (targetTriple.isPPC())
    targetCharacteristics.set_isPPC(true);

  // TODO: use target machine data layout to set-up the target characteristics
  // type size and alignment info.
}

/// Create a target machine that is at least sufficient to get data-layout
/// information required by flang semantics and lowering. Note that it may not
/// contain all the CPU feature information to get optimized assembly generation
/// from LLVM IR. Drivers that needs to generate assembly from LLVM IR should
/// create a target machine according to their specific options.
[[maybe_unused]] inline static std::unique_ptr<llvm::TargetMachine>
createTargetMachine(llvm::StringRef targetTriple, std::string &error) {
  std::string triple{targetTriple};
  if (triple.empty())
    triple = llvm::sys::getDefaultTargetTriple();

  const llvm::Target *theTarget =
      llvm::TargetRegistry::lookupTarget(triple, error);
  if (!theTarget)
    return nullptr;
  return std::unique_ptr<llvm::TargetMachine>{
      theTarget->createTargetMachine(triple, /*CPU=*/"",
          /*Features=*/"", llvm::TargetOptions(),
          /*Reloc::Model=*/std::nullopt)};
}
} // namespace Fortran::tools

#endif // FORTRAN_TOOLS_TARGET_SETUP_H
