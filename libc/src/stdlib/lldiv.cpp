//===-- Implementation of lldiv -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/lldiv.h"
#include "src/__support/common.h"
#include "src/__support/integer_operations.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(lldiv_t, lldiv, (long long x, long long y)) {
  lldiv_t res;
  integerRemQuo(x, y, res.quot, res.rem);
  return res;
}

} // namespace __llvm_libc
