//===-- Implementation of localtime for baremetal -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/localtime.h"
#include "hdr/time_macros.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

extern "C" bool __llvm_libc_localtime_utc(struct timespec *ts);

LLVM_LIBC_FUNCTION(int, localtime, (struct timespec * ts, int base)) {
  (void)ts;
  return base;
}

} // namespace LIBC_NAMESPACE_DECL
