//===-- Implementation of localtime_r for baremetal -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/localtime_r.h"
#include "hdr/types/struct_tm.h"
#include "hdr/types/time_t.h"
#include "src/__support/macros/null_check.h"
#include "src/time/time_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(struct tm *, localtime_r,
                   (const time_t *timer, struct tm *buf)) {
  LIBC_CRASH_ON_NULLPTR(timer);

  return time_utils::localtime_internal(timer, buf);
}

} // namespace LIBC_NAMESPACE_DECL
