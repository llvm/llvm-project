//===-- Implementation of strftime function -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/strftime.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/errno/libc_errno.h"
#include "src/time/time_utils.h"

namespace LIBC_NAMESPACE_DECL {

using LIBC_NAMESPACE::time_utils::TimeConstants;

LLVM_LIBC_FUNCTION(size_t, strftime,
                   (char *__restrict, size_t, const char *__restrict,
                    const struct tm *)) {
  // TODO: Implement this for the default locale.
  return -1;
}

} // namespace LIBC_NAMESPACE_DECL
