//===-- Implementation of fopen_s -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fopen_s.h"
#include "src/__support/annex_k/helper_macros.h"
#include "src/__support/macros/config.h"
#include "src/__support/stdio/fopen.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(errno_t, fopen_s,
                   (FILE *__restrict *__restrict streamptr,
                    const char *__restrict filename,
                    const char *__restrict mode)) {
  _CONSTRAINT_VIOLATION_IF(streamptr == 0, EINVAL, EINVAL);
  _CONSTRAINT_VIOLATION_CLEANUP_IF(!mode || !filename, *streamptr = nullptr,
                                   EINVAL, EINVAL);

  FILE *ret = nullptr;

  if (mode[0] == 'u') {
    ret = stdio_internal::fopen(filename, mode + 1);
    if (!ret) {
      *streamptr = nullptr;
      return -1;
    }
  } else {
    ret = stdio_internal::fopen(filename, mode);
    if (!ret) {
      *streamptr = nullptr;
      return -1;
    }
  }

  *streamptr = ret;
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
