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
  // matches in size.
  static_assert(sizeof(int) == sizeof(fexcept_t),
                "sizeof(fexcept_t) != sizeof(int)");
  int excepts_to_set = *reinterpret_cast<const int *>(flagp) & excepts;
  return fputil::setExcept(excepts_to_set);
}

} // namespace __llvm_libc
