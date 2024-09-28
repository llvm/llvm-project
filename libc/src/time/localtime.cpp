//===-- Implementation of localtime function ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/localtime.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/time/time_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(struct tm *, localtime, (const time_t *t_ptr)) {
  if (t_ptr == nullptr || *t_ptr > cpp::numeric_limits<int32_t>::max()) {
      return nullptr;
  }

  return time_utils::localtime(t_ptr);
}

} // namespace LIBC_NAMESPACE_DECL
