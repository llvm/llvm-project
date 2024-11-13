//===-- Tools/TargetSetup.h ------------------------------------- *-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_TOOLS_TARGET_SETUP_H
#define FORTRAN_TOOLS_TARGET_SETUP_H

#include "flang/Common/float128.h"
#include "flang/Evaluate/target.h"
#include "flang/Frontend/TargetOptions.h"
#include "llvm/Target/TargetMachine.h"
#include <cfloat>

namespace Fortran::tools {

[[maybe_unused]] inline static void setUpTargetCharacteristics(
    Fortran::evaluate::TargetCharacteristics &targetCharacteristics,
    const llvm::TargetMachine &targetMachine,
    const Fortran::frontend::TargetOptions &targetOptions,
    const std::string &compilerVersion, const std::string &compilerOptions) {

  const llvm::Triple &targetTriple{targetMachine.getTargetTriple()};
  // FIXME: Handle real(3) ?
  if (targetTriple.getArch() != llvm::Triple::ArchType::x86_64) {
    targetCharacteristics.DisableType(
        Fortran::common::TypeCategory::Real, /*kind=*/10);
  }

  // Figure out if we can support F128: see
  // flang/runtime/Float128Math/math-entries.h
  // TODO: this should be taken from TargetInfo::getLongDoubleFormat to support
  // cross-compilation
#ifdef FLANG_RUNTIME_F128_MATH_LIB
  // we can use libquadmath wrappers
  constexpr bool f128Support = true;
#elif HAS_LDBL128
  // we can use libm wrappers
  constexpr bool f128Support = true;
#else
  constexpr bool f128Support = false;
#endif

  if constexpr (!f128Support)
    targetCharacteristics.DisableType(Fortran::common::TypeCategory::Real, 16);

  for (auto realKind : targetOptions.disabledRealKinds)
    targetCharacteristics.DisableType(common::TypeCategory::Real, realKind);

  for (auto intKind : targetOptions.disabledIntegerKinds)
    targetCharacteristics.DisableType(common::TypeCategory::Integer, intKind);

  targetCharacteristics.set_compilerOptionsString(compilerOptions)
      .set_compilerVersionString(compilerVersion);

  if (targetTriple.isPPC())
    targetCharacteristics.set_isPPC(true);

  if (targetTriple.isOSWindows())
    targetCharacteristics.set_isOSWindows(true);

  // TODO: use target machine data layout to set-up the target characteristics
  // type size and alignment info.
}

} // namespace Fortran::tools

#endif // FORTRAN_TOOLS_TARGET_SETUP_H
