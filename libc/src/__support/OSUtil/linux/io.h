//===-------------- Linux implementation of IO utils ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_IO_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_IO_H

#include "hdr/errno_macros.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/macros/config.h"
#include "syscall.h"     // For internal syscall function.
#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {

LIBC_INLINE void write_to_stderr(cpp::string_view msg) {
  size_t written = 0;
  const char *msg_data = msg.data();
  while (written != msg.size()) {
    auto delta = LIBC_NAMESPACE::syscall_impl<long>(SYS_write, 2 /* stderr */,
                                                    msg_data, msg.size());
    // If the write syscall was interrupted, try again. Otherwise, this is an
    // error routine, we do not handle further errors.
    if (delta == -EINTR)
      continue;
    if (delta < 0)
      return;
    written += delta;
    msg_data += delta;
  }
}

template <size_t N>
LIBC_INLINE void write_all_to_stderr(const cpp::string_view (&msgs)[N]) {
  struct IOVec {
    const char *base;
    size_t len;
  } iovs[N];

  size_t total_len = 0;
  for (size_t i = 0; i < N; ++i) {
    iovs[i].base = msgs[i].data();
    iovs[i].len = msgs[i].size();
    total_len += msgs[i].size();
  }

  size_t written = 0;
  while (written != total_len) {
    auto delta =
        LIBC_NAMESPACE::syscall_impl<long>(SYS_writev, 2 /* stderr */, iovs, N);

    // If the write syscall was interrupted, try again. Otherwise, this is an
    // error routine, we do not handle further errors.
    if (delta == -EINTR)
      continue;
    if (delta < 0)
      return;

    auto udelta = static_cast<size_t>(delta);
    written += udelta;
    for (size_t i = 0; i < N; ++i) {
      if (udelta == 0)
        break;
      auto change = udelta < iovs[i].len ? udelta : iovs[i].len;
      udelta -= change;
      iovs[i].base += change;
      iovs[i].len -= change;
    }
  }
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_IO_H
