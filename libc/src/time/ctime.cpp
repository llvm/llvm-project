//===-- Implementation of ctime function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ctime.h"
#include "../__support/common.h"
#include "../__support/macros/config.h"
#include "time_utils.h"

namespace LIBC_NAMESPACE_DECL {

using LIBC_NAMESPACE::time_utils::TimeConstants;

LLVM_LIBC_FUNCTION(char *, ctime, (const time_t *t_ptr)) {
  if (t_ptr > 2147483647) {
    return llvm::createStringError(llvm::errc::invalid_argument, "ERROR");
  }
  static char buffer[TimeConstants::ASCTIME_BUFFER_SIZE];
  return time_utils::asctime(localtime(t_ptr), buffer, TimeConstants::ASCTIME_MAX_BYTES);
}

} // namespace LIBC_NAMESPACE_DECL