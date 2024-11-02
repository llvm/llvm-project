//===-- Linux implementation of cfsetospeed -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/termios/cfsetospeed.h"

#include "src/__support/common.h"

#include <errno.h>
#include <termios.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, cfsetospeed, (struct termios * t, speed_t speed)) {
  constexpr speed_t NOT_SPEED_MASK = ~speed_t(CBAUD);
  // A speed value is valid only if it is equal to one of the B<NN+> values.
  if (t == nullptr || ((speed & NOT_SPEED_MASK) != 0)) {
    errno = EINVAL;
    return -1;
  }

  t->c_cflag = (t->c_cflag & NOT_SPEED_MASK) | speed;
  t->c_ospeed = speed;
  return 0;
}

} // namespace __llvm_libc
