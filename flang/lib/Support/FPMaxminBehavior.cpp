//===--- Support/FPMaxminBehavior.cpp - Parse FP max/min behavior ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Support/FPMaxminBehavior.h"
#include "llvm/ADT/StringSwitch.h"

namespace Fortran::common {

FPMaxminBehavior parseFPMaxminBehavior(llvm::StringRef value) {
  return llvm::StringSwitch<FPMaxminBehavior>(value)
      .Case("legacy", FPMaxminBehavior::Legacy)
      .Case("portable", FPMaxminBehavior::Portable)
      .Case("extremum", FPMaxminBehavior::Extremum)
      .Case("extremenum", FPMaxminBehavior::ExtremeNum)
      .Default(FPMaxminBehavior::Legacy);
}

} // namespace Fortran::common
