//===-- Implementation of fegetexceptflag function ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fenv/fegetexceptflag.h"
#include "src/__support/common.h"
#include "utils/FPUtil/FEnv.h"

#include <fenv.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, fegetexceptflag, (fexcept_t * flagp, int excepts)) {
  // Since the return type of fetestexcept is int, we ensure that fexcept_t
  // matches in size.
  static_assert(sizeof(int) == sizeof(fexcept_t),
                "sizeof(fexcept_t) != sizeof(int)");
  *reinterpret_cast<int *>(flagp) = fputil::testExcept(FE_ALL_EXCEPT) & excepts;
  return 0;
}

} // namespace __llvm_libc
