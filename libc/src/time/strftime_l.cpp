//===-- Implementation of strftime_l function -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/strftime_l.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/errno/libc_errno.h"
#include "src/time/time_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(size_t, strftime_l,
                   (char *__restrict, size_t, const char *__restrict,
                    const struct tm *, locale_t)) {
  // TODO: Implement this for the default locale.
  return -1;
}

} // namespace LIBC_NAMESPACE_DECL
