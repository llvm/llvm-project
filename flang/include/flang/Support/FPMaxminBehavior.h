//===- Support/FPMaxminBehavior.h - FP max/min behavior option --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Shared definition of FP max/min behavior for max/min and [max|min][loc|val].
/// Used by CodeGenOptions, LoweringOptions, and other components.
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SUPPORT_FPMAXMINBEHAVIOR_H_
#define FORTRAN_SUPPORT_FPMAXMINBEHAVIOR_H_

#include "llvm/ADT/StringRef.h"

namespace Fortran::common {

/// Control for max/min and [max|min][loc|val] lowering, constant folding, and
/// related behavior. Legacy: current Flang behavior (always cmp+select).
/// Portable: same as Legacy but may use arith.maxnumf under
/// '-fno-signed-zeros -fno-honor-nans'.
/// Extremum: arith.maximumf/minimumf
/// ExtremeNum: arith.maxnumf/minnumf.
/// Legacy is transitional and will eventually be replaced by Portable.
enum class FPMaxminBehavior : unsigned {
  Legacy,
  Portable,
  Extremum,
  ExtremeNum,
};

/// Parse -ffp-maxmin-behavior= value. Triggers llvm_unreachable
/// for unknown strings.
FPMaxminBehavior parseFPMaxminBehavior(llvm::StringRef value);

} // namespace Fortran::common

#endif // FORTRAN_SUPPORT_FPMAXMINBEHAVIOR_H_
