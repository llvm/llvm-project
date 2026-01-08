//===-- Implementation of localtime for baremetal -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/localtime.h"
#include "hdr/types/struct_tm.h"
#include "hdr/types/time_t.h"
#include "src/time/time_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(struct tm *, localtime, (time_t *timer)) {
  static struct tm tm_out;

  return time_utils::localtime_internal(timer, &tm_out);
}

} // namespace LIBC_NAMESPACE_DECL
