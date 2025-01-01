//===-- Implementation of ctime_r function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ctime_r.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "time_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(char *, ctime_r, (const time_t *t_ptr, char *buffer)) {
  static struct tm tm_out;

  if (t_ptr == nullptr || buffer == nullptr ||
      *t_ptr > cpp::numeric_limits<int32_t>::max()) {
    return nullptr;
  }

  return time_utils::asctime(time_utils::localtime(t_ptr), buffer,
                             TimeConstants::ASCTIME_MAX_BYTES);
}

} // namespace LIBC_NAMESPACE_DECL
