//===-- Darwin implementation of sigfillset --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/signal/sigfillset.h"
#include "hdr/types/sigset_t.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, sigfillset, (sigset_t * set)) {
  if (!set) {
    libc_errno = EINVAL;
    return -1;
  }
  // Darwin sigset_t is 32-bit, but the defined signal range here is 1..31.
  set->__signals[0] = 0x7FFFFFFFU;
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
