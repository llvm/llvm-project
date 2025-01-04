//===-- Implementation of localtime_s function ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/localtime_s.h"
#include "src/time/time_utils.h"

namespace LIBC_NAMESPACE_DECL {

// windows only, implemented in gnu/linux for compatibility reasons
LLVM_LIBC_FUNCTION(int, localtime_s, (const time_t *timer, struct tm *input)) {
  if (timer == nullptr) {
    return nullptr;
  }

  return time_utils::localtime_s_internal(timer, input);
}

} // namespace LIBC_NAMESPACE_DECL
