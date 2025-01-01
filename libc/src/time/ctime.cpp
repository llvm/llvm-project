//===-- Implementation of ctime function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ctime.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "time_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(char *, ctime, (const time_t *t_ptr)) {
  static struct tm tm_out;

  if (t_ptr == nullptr || *t_ptr > cpp::numeric_limits<int32_t>::max()) {
    return nullptr;
  }

  static char buffer[TimeConstants::ASCTIME_BUFFER_SIZE];
  return time_utils::asctime(time_utils::localtime_internal(t_ptr, &tm_out),
                             buffer, TimeConstants::ASCTIME_MAX_BYTES);
}

} // namespace LIBC_NAMESPACE_DECL
