//===-- Implementation of memcpy ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/memcpy.h"
#include "src/__support/common.h"
#include "src/string/memcpy_arch_specific.h"

namespace __llvm_libc {

void *LLVM_LIBC_ENTRYPOINT(memcpy)(void *__restrict dst,
                                   const void *__restrict src, size_t size) {
  memcpy_no_return(reinterpret_cast<char *>(dst),
                   reinterpret_cast<const char *>(src), size);
  return dst;
}

} // namespace __llvm_libc
