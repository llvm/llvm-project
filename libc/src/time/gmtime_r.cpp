//===-- Implementation of gmtime_r function -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/gmtime_r.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/time/time_utils.h"
#include "src/time/timezone.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(struct tm *, gmtime_r,
                   (const time_t *timer, struct tm *result)) {
  return time_utils::gmtime_internal(timer, result);
}

} // namespace LIBC_NAMESPACE_DECL
