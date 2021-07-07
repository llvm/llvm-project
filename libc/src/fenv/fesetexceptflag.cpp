//===-- Implementation of fesetexceptflag function ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fenv/fesetexceptflag.h"
#include "src/__support/common.h"
#include "utils/FPUtil/FEnv.h"

#include <fenv.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, fesetexceptflag,
                   (const fexcept_t *flagp, int excepts)) {
  // Since the return type of fetestexcept is int, we ensure that fexcept_t
  // can fit in int type.
  static_assert(sizeof(int) >= sizeof(fexcept_t),
                "fexcept_t value cannot fit in an int value.");
  int excepts_to_set = static_cast<const int>(*flagp) & excepts;
  fputil::clearExcept(FE_ALL_EXCEPT);
  return fputil::setExcept(excepts_to_set);
}

} // namespace __llvm_libc
