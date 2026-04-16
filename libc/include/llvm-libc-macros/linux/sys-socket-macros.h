//===-- Definition of macros from sys/socket.h ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_MACROS_LINUX_SYS_SOCKET_MACROS_H
#define LLVM_LIBC_MACROS_LINUX_SYS_SOCKET_MACROS_H

// IEEE Std 1003.1-2017 - basedefs/sys_socket.h.html
// Macro values come from the Linux syscall interface.

#define AF_UNSPEC 0 // Unspecified
#define AF_UNIX 1   // Unix domain sockets
#define AF_LOCAL 1  // POSIX name for AF_UNIX
#define AF_INET 2   // Internet IPv4 Protocol
#define AF_INET6 10 // IP version 6

#define SOCK_STREAM 1
#define SOCK_DGRAM 2
#define SOCK_RAW 3
#define SOCK_RDM 4
#define SOCK_SEQPACKET 5
#define SOCK_PACKET 10

#define SOL_SOCKET 1

#define SO_DEBUG 1
#define SO_REUSEADDR 2
#define SO_TYPE 3
#define SO_ERROR 4
#define SO_DONTROUTE 5
#define SO_BROADCAST 6
#define SO_SNDBUF 7
#define SO_RCVBUF 8
#define SO_KEEPALIVE 9
#define SO_OOBINLINE 10
#define SO_NO_CHECK 11
#define SO_PRIORITY 12
#define SO_LINGER 13
#define SO_BSDCOMPAT 14
#define SO_REUSEPORT 15

#endif // LLVM_LIBC_MACROS_LINUX_SYS_SOCKET_MACROS_H
