//===-- Implementation of ctime_s function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __STDC_WANT_LIB_EXT1__ 1

#include "ctime_s.h"
#include "hdr/errno_macros.h"
#include "hdr/types/errno_t.h"
#include "hdr/types/rsize_t.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "time_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(errno_t, ctime_s,
                   (char *buffer, rsize_t buffer_size, const time_t *t_ptr)) {
  // TODO (https://github.com/llvm/llvm-project/issues/115907): invoke
  // constraint handler
  if (buffer == nullptr || t_ptr == nullptr)
    return EINVAL;

  if (buffer_size < time_constants::ASCTIME_MAX_BYTES ||
      buffer_size > RSIZE_MAX) {
    buffer[0] = '\0';
    return ERANGE;
  }

  if (*t_ptr > cpp::numeric_limits<int32_t>::max())
    return EINVAL;

  if (time_utils::asctime(time_utils::localtime(t_ptr), buffer, buffer_size) ==
      nullptr)
    return EINVAL;

  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
