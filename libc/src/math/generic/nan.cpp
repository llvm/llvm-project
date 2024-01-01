//===-- Implementation of nan function ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/nan.h"
#include "src/__support/common.h"
#include "src/__support/str_to_float.h"
#include "src/errno/libc_errno.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(double, nan, (const char *arg)) {
  const char *fp_str = internal::nan_str_to_floatingpoint_str(arg);
  auto result = internal::strtofloatingpoint<double>(fp_str);

  if (result.has_error())
    libc_errno = result.error;

  return result.value;
}

} // namespace LIBC_NAMESPACE
