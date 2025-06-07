//===-- Implementation of localtime for baremetal -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/localtime.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(struct tm *, localtime, (time_t *timer)) {
  return time_utils::localtime_internal(timer, buf);
}

} // namespace LIBC_NAMESPACE_DECL
