//===-- Implementation of timespec_get for baremetal ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/timespec_get.h"
#include "hdr/time_macros.h"
#include "src/__support/OSUtil/baremetal/time.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, timespec_get, (struct timespec * ts, int base)) {
  if (base != TIME_UTC) {
    return 0;
  }

  if (!internal::timespec_get_utc(ts)) {
    return 0;
  }
  return base;
}

} // namespace LIBC_NAMESPACE_DECL
