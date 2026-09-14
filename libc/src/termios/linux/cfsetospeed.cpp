//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of cfsetospeed.
///
//===----------------------------------------------------------------------===//

#include "src/termios/cfsetospeed.h"
#include "hdr/types/struct_termios.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/null_check.h"
#include "src/termios/linux/speed_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, cfsetospeed, (termios * t, speed_t speed)) {
  LIBC_CRASH_ON_NULLPTR(t);
  constexpr speed_t NOT_SPEED_MASK = ~static_cast<speed_t>(CBAUD);
  speed_t encoded = encode_speed(speed);
  // A speed value is valid only if it is equal to one of the B<NN+> values.
  if ((encoded & NOT_SPEED_MASK) != 0) {
    libc_errno = EINVAL;
    return -1;
  }

  t->c_cflag = (t->c_cflag & NOT_SPEED_MASK) | encoded;
  t->c_ospeed = speed;
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
