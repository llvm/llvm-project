/* ID for functions called via socketcall system call.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef _SYS_SOCKETCALL_H
#define _SYS_SOCKETCALL_H	1

#include <sysdep.h>

/* Define unique numbers for the operations permitted on socket.  Linux
   uses a single system call for all these functions.  The relevant code
   file is /usr/include/linux/net.h.
   We cannot use an enum here because the values are used in assembler
   code.  */

#define SOCKOP_invalid		-1
#define SOCKOP_socket		1
#define SOCKOP_bind		2
#define SOCKOP_connect		3
#define SOCKOP_listen		4
#define SOCKOP_accept		5
#define SOCKOP_getsockname	6
#define SOCKOP_getpeername	7
#define SOCKOP_socketpair	8
#define SOCKOP_send		9
#define SOCKOP_recv		10
#define SOCKOP_sendto		11
#define SOCKOP_recvfrom		12
#define SOCKOP_shutdown		13
#define SOCKOP_setsockopt	14
#define SOCKOP_getsockopt	15
#define SOCKOP_sendmsg		16
#define SOCKOP_recvmsg		17
#define SOCKOP_accept4		18
#define SOCKOP_recvmmsg		19
#define SOCKOP_sendmmsg		20

#define __SOCKETCALL1(name, a1) \
  INLINE_SYSCALL (socketcall, 2, name, \
     ((long int [1]) { (long int) (a1) }))
#define __SOCKETCALL2(name, a1, a2) \
  INLINE_SYSCALL (socketcall, 2, name, \
     ((long int [2]) { (long int) (a1), (long int) (a2) }))
#define __SOCKETCALL3(name, a1, a2, a3) \
  INLINE_SYSCALL (socketcall, 2, name, \
     ((long int [3]) { (long int) (a1), (long int) (a2), (long int) (a3) }))
#define __SOCKETCALL4(name, a1, a2, a3, a4) \
  INLINE_SYSCALL (socketcall, 2, name, \
     ((long int [4]) { (long int) (a1), (long int) (a2), (long int) (a3), \
                       (long int) (a4) }))
#define __SOCKETCALL5(name, a1, a2, a3, a4, a5) \
  INLINE_SYSCALL (socketcall, 2, name, \
     ((long int [5]) { (long int) (a1), (long int) (a2), (long int) (a3), \
                       (long int) (a4), (long int) (a5) }))
#define __SOCKETCALL6(name, a1, a2, a3, a4, a5, a6) \
  INLINE_SYSCALL (socketcall, 2, name, \
     ((long int [6]) { (long int) (a1), (long int) (a2), (long int) (a3), \
                       (long int) (a4), (long int) (a5), (long int) (a6) }))

#define __SOCKETCALL_NARGS_X(a,b,c,d,e,f,g,h,n,...) n
#define __SOCKETCALL_NARGS(...) \
  __SOCKETCALL_NARGS_X (__VA_ARGS__,7,6,5,4,3,2,1,0,)
#define __SOCKETCALL_CONCAT_X(a,b)     a##b
#define __SOCKETCALL_CONCAT(a,b)       __SOCKETCALL_CONCAT_X (a, b)
#define __SOCKETCALL_DISP(b,...) \
  __SOCKETCALL_CONCAT (b,__SOCKETCALL_NARGS(__VA_ARGS__))(__VA_ARGS__)

#define __SOCKETCALL(...) __SOCKETCALL_DISP (__SOCKETCALL, __VA_ARGS__)


#define SOCKETCALL(name, args...)					\
  ({									\
    long int sc_ret = __SOCKETCALL (SOCKOP_##name, args);		\
    sc_ret;								\
  })


#define SOCKETCALL_CANCEL(name, args...)				\
  ({									\
    int oldtype = LIBC_CANCEL_ASYNC ();					\
    long int sc_ret = __SOCKETCALL (SOCKOP_##name, args);		\
    LIBC_CANCEL_RESET (oldtype);					\
    sc_ret;								\
  })


#endif /* sys/socketcall.h */
