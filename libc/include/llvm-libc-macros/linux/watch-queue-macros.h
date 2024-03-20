//===-- Definition of macros from watch-queue.h ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// References
// https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/include/uapi/linux/watch_queue.h
// https://kernelnewbies.org/Linux_5.8#Core_.28various.29
// https://docs.kernel.org/core-api/watch_queue.html

#ifndef LLVM_LIBC_MACROS_LINUX_WATCH_QUEUE_MACROS_H
#define LLVM_LIBC_MACROS_LINUX_WATCH_QUEUE_MACROS_H

#define O_NOTIFICATION_PIPE                                                    \
  O_EXCL /* Parameter to pipe2() selecting notification pipe */

#endif // LLVM_LIBC_MACROS_LINUX_WATCH_QUEUE_MACROS_H
