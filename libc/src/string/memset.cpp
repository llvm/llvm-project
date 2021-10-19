//===-- Implementation of memset ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/memset.h"
#include "src/__support/common.h"
#include "src/string/memory_utils/memset_utils.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(void *, memset, (void *dst, int value, size_t count)) {
  GeneralPurposeMemset(reinterpret_cast<char *>(dst),
                       static_cast<unsigned char>(value), count);
  return dst;
}

} // namespace __llvm_libc
