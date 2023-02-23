//===-- Definition of macros from sys/socket.h ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_MACROS_LINUX_SYS_SOCKET_MACROS_H
#define __LLVM_LIBC_MACROS_LINUX_SYS_SOCKET_MACROS_H

// IEEE Std 1003.1-2017 - basedefs/sys_socket.h.html
// Macro values come from the Linux syscall interface.

#define AF_UNSPEC 0 // Unspecified
#define AF_UNIX 1   // Unix domain sockets
#define AF_LOCAL 1  // POSIX name for AF_UNIX
#define AF_INET 2   // Internet IPv4 Protocol
#define AF_INET6 10 // IP version 6

#endif // __LLVM_LIBC_MACROS_LINUX_SYS_SOCKET_MACROS_H
