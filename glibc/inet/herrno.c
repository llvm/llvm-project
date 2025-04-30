/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
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

#include <features.h>
#include <netdb.h>
#undef h_errno

#include <tls.h>

/* We need to have the error status variable of the resolver
   accessible in the libc.  */

__thread int __h_errno;
extern __thread int __libc_h_errno __attribute__ ((alias ("__h_errno")))
  attribute_hidden;
#define h_errno __libc_h_errno
