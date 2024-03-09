//===-- Implementation of strfroml ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/strfroml.h"
#include "src/__support/common.h"
#include "src/__support/str_to_float.h"
#include "src/errno/libc_errno.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, strfroml,
                   (const char *__restrict s, size_t n,
                    const char *__restrict format, long double fp)) {
  (void)s;
  (void)n;
  (void)format;
  (void)fp;
  return 0;
}

} // namespace LIBC_NAMESPACE
