//===-- Implementation of ctime_s function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ctime_s.h"
#include "hdr/errno_macros.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "time_utils.h"

namespace LIBC_NAMESPACE_DECL {

using LIBC_NAMESPACE::time_utils::TimeConstants;

LLVM_LIBC_FUNCTION(int, ctime_s,
                   (char *buffer, size_t buffer_size, const time_t *t_ptr)) {
  if (t_ptr == nullptr || buffer == nullptr ||
      *time > cpp::numeric_limits<int32_t>::max()) {
    return EINVAL;
  }

  if (buffer_size < TimeConstants::ASCTIME_MAX_BYTES) {
    return ERANGE;
  }

  if (time_utils::asctime(time_utils::localtime(t_ptr), buffer, buffer_size) ==
      nullptr) {
    return EINVAL;
  }

  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
