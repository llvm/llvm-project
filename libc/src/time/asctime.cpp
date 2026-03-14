//===-- Implementation of asctime function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/asctime.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/time/time_constants.h"
#include "src/time/time_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(char *, asctime, (const struct tm *timeptr)) {
  static char buffer[time_constants::ASCTIME_BUFFER_SIZE];
  return time_utils::asctime(timeptr, buffer,
                             time_constants::ASCTIME_MAX_BYTES);
}

} // namespace LIBC_NAMESPACE_DECL
