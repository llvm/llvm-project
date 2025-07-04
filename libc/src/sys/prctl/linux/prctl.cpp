//===---------- Linux implementation of the prctl function ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/prctl/prctl.h"

#include "src/__support/OSUtil/prctl.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, prctl,
                   (int option, unsigned long arg2, unsigned long arg3,
                    unsigned long arg4, unsigned long arg5)) {
  auto result = internal::prctl(option, arg2, arg3, arg4, arg5);

  if (!result.has_value()) {
    libc_errno = result.error();
    return -1;
  }

  return result.value();
}

} // namespace LIBC_NAMESPACE_DECL
