//===-- Implementation of ctime_r function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/ctime_r.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/time/time_utils.h"

namespace LIBC_NAMESPACE_DECL {

using LIBC_NAMESPACE::time_utils::TimeConstants;

LLVM_LIBC_FUNCTION(char *, ctime_r, (const time_t *t_ptr, char *buffer)) {
  return time_utils::asctime(localtime(t_ptr), buffer, TimeConstants::CTIME_MAX_BYTES);
}

} // namespace LIBC_NAMESPACE_DECL
