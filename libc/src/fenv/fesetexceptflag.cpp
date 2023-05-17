//===-- Implementation of fesetexceptflag function ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fenv/fesetexceptflag.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/common.h"

#include <fenv.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, fesetexceptflag,
                   (const fexcept_t *flagp, int excepts)) {
  // Since the return type of fetestexcept is int, we ensure that fexcept_t
  // can fit in int type.
  static_assert(sizeof(int) >= sizeof(fexcept_t),
                "fexcept_t value cannot fit in an int value.");
  int excepts_to_set = static_cast<int>(*flagp) & excepts;
  fputil::clear_except(FE_ALL_EXCEPT);
  return fputil::set_except(excepts_to_set);
}

} // namespace __llvm_libc
