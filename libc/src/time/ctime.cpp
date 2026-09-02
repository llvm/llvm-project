//===-- Implementation of ctime function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/ctime.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/time/time_constants.h"
#include "src/time/time_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(char *, ctime, (const time_t *t_ptr)) {
  LIBC_CRASH_ON_NULLPTR(t_ptr);

  auto lt_res = time_utils::localtime(t_ptr);
  if (!lt_res) {
    libc_errno = lt_res.error();
    return nullptr;
  }

  static char buffer[time_constants::ASCTIME_BUFFER_SIZE];
  auto res = time_utils::asctime(lt_res.value(), buffer,
                                 time_constants::ASCTIME_MAX_BYTES);
  if (!res) {
    libc_errno = res.error();
    return nullptr;
  }
  return res.value();
}

} // namespace LIBC_NAMESPACE_DECL
