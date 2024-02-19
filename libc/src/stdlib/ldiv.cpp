//===-- Implementation of ldiv --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/ldiv.h"
#include "src/__support/common.h"
#include "src/__support/integer_operations.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(ldiv_t, ldiv, (long x, long y)) {
  ldiv_t res;
  integer_rem_quo(x, y, res.quot, res.rem);
  return res;
}

} // namespace LIBC_NAMESPACE
