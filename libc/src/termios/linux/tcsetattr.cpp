//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of tcsetattr.
///
//===----------------------------------------------------------------------===//

#include "src/termios/tcsetattr.h"
#include "hdr/termios_macros.h"
#include "hdr/types/struct_termios.h"
#include "src/__support/CPP/algorithm.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/ioctl.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/null_check.h"
#include "src/termios/linux/kernel_termios.h"
#include "src/termios/linux/speed_utils.h"

#include <asm/ioctls.h> // Safe to include without the risk of name pollution.

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, tcsetattr, (int fd, int actions, const termios *t)) {
  LIBC_CRASH_ON_NULLPTR(t);
  struct kernel_termios kt;
  long cmd;

  switch (actions) {
  case TCSANOW:
    cmd = TCSETS;
    break;
  case TCSADRAIN:
    cmd = TCSETSW;
    break;
  case TCSAFLUSH:
    cmd = TCSETSF;
    break;
  default:
    libc_errno = EINVAL;
    return -1;
  }

  kt.c_iflag = t->c_iflag;
  kt.c_oflag = t->c_oflag;

  speed_t ospeed = encode_speed(t->c_ospeed);
  speed_t ispeed = t->c_ispeed == 0 ? ospeed : encode_speed(t->c_ispeed);

  constexpr speed_t NOT_SPEED_MASK = ~static_cast<speed_t>(CBAUD | CIBAUD);
  kt.c_cflag = (t->c_cflag & NOT_SPEED_MASK) | ospeed | (ispeed << 16);

  kt.c_lflag = t->c_lflag;
  kt.c_line = t->c_line;

  size_t nccs = cpp::min(KERNEL_NCCS, static_cast<size_t>(NCCS));
  for (size_t i = 0; i < nccs; ++i)
    kt.c_cc[i] = t->c_cc[i];
  if (nccs < KERNEL_NCCS) {
    for (size_t i = nccs; i < KERNEL_NCCS; ++i)
      kt.c_cc[i] = 0;
  }

  auto ret = linux_syscalls::ioctl(fd, cmd, &kt);
  if (!ret.has_value()) {
    libc_errno = ret.error();
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
