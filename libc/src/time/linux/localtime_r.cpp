//===-- Implementation of localtime_r function ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/localtime_r.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/time/time_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(struct tm *, localtime_r,
                   (const time_t *timer, struct tm *buf)) {
  return time_utils::localtime_internal(timer, buf);
}

} // namespace LIBC_NAMESPACE_DECL
