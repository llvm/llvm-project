//===-- Implementation of memcmp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/memcmp.h"
#include "src/__support/common.h"
#include <stddef.h> // size_t

namespace __llvm_libc {

// TODO: It is a simple implementation, an optimized version is preparing.
LLVM_LIBC_FUNCTION(int, memcmp,
                   (const void *lhs, const void *rhs, size_t count)) {
  const unsigned char *_lhs = reinterpret_cast<const unsigned char *>(lhs);
  const unsigned char *_rhs = reinterpret_cast<const unsigned char *>(rhs);
  for (size_t i = 0; i < count; ++i)
    if (_lhs[i] != _rhs[i])
      return _lhs[i] - _rhs[i];
  // count is 0 or _lhs and _rhs are the same.
  return 0;
}

} // namespace __llvm_libc
