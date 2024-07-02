//===-- Implementation of strftime function -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/strftime.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {


LLVM_LIBC_FUNCTION(size_t, strftime, (char*__restrict, size_t,
                       const char *__restrict,
                       const struct tm *__restrict)) {
  return 0;
}

} // namespace LIBC_NAMESPACE
