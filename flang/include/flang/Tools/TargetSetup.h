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
#include "llvm/Target/TargetMachine.h"

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

} // namespace Fortran::tools

#endif // FORTRAN_TOOLS_TARGET_SETUP_H
