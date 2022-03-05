//===-- Implementation of feclearexcept function --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fenv/feclearexcept.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/common.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, feclearexcept, (int e)) {
  return fputil::clear_except(e);
}

} // namespace __llvm_libc
