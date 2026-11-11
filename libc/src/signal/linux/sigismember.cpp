//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of sigismember function.
///
//===----------------------------------------------------------------------===//

#include "src/signal/sigismember.h"

#include "hdr/errno_macros.h"
#include "hdr/types/sigset_t.h"
#include "src/__support/OSUtil/linux/sigmask.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, sigismember, (const sigset_t *set, int signum)) {
  if (set == nullptr) {
    libc_errno = EINVAL;
    return -1;
  }
  auto res = signals::is_member(*set, signum);
  if (!res) {
    libc_errno = res.error();
    return -1;
  }
  return res.value();
}

} // namespace LIBC_NAMESPACE_DECL
