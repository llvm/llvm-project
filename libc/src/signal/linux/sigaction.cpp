//===-- Linux implementation of sigaction ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/signal/sigaction.h"

#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/signal/linux/signal_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, sigaction,
                   (int signal, const struct sigaction *__restrict libc_new,
                    struct sigaction *__restrict libc_old)) {
  ErrorOr<int> ret = do_sigaction(signal, libc_new, libc_old);
  if (ret)
    return ret.value();

  libc_errno = ret.error();
  return -1;
}

} // namespace LIBC_NAMESPACE_DECL
