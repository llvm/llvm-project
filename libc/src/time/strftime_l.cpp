//===-- Implementation of strftime_l function -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/strftime_l.h"
#include "src/time/strftime.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {


LLVM_LIBC_FUNCTION(size_t, strftime_l, (char*__restrict arg1, size_t arg2,
                       const char *__restrict arg3,
                       const struct tm *__restrict arg4, locale_t)) {
  return LIBC_NAMESPACE::strftime(arg1, arg2, arg3, arg4);
}

} // namespace LIBC_NAMESPACE
