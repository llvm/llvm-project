/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
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

/* Linux version.  */

#ifndef _BITS_ERRQUEUE_H
#define _BITS_ERRQUEUE_H  1

#include <sys/types.h>
#include <sys/socket.h>

struct sock_extended_err
  {
    uint32_t ee_errno;
    uint8_t ee_origin;
    uint8_t ee_type;
    uint8_t ee_code;
    uint8_t ee_pad;
    uint32_t ee_info;
    uint32_t ee_data;
  };

#define SO_EE_ORIGIN_NONE  0
#define SO_EE_ORIGIN_LOCAL 1
#define SO_EE_ORIGIN_ICMP  2
#define SO_EE_ORIGIN_ICMP6 3

#define SO_EE_OFFENDER(see)	\
  ((struct sockaddr *)(((struct sock_extended_err)(see))+1))

#endif /* bits/errqueue.h */
