//===-- Implementation of bcmp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/bcmp.h"
#include "src/__support/common.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, bcmp,
                   (const void *lhs, const void *rhs, size_t count)) {
  const unsigned char *_lhs = reinterpret_cast<const unsigned char *>(lhs);
  const unsigned char *_rhs = reinterpret_cast<const unsigned char *>(rhs);
  for (size_t i = 0; i < count; ++i) {
    if (_lhs[i] != _rhs[i]) {
      return 1;
    }
  }
  // count is 0 or _lhs and _rhs are the same.
  return 0;
}

} // namespace __llvm_libc
