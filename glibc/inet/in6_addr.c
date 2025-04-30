/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Philip Blundell <pjb27@cam.ac.uk>, 1997.

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

#include <netinet/in.h>

const struct in6_addr __in6addr_any =
{ { { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 } } };
libc_hidden_data_def (__in6addr_any)
weak_alias (__in6addr_any, in6addr_any)
libc_hidden_data_weak (in6addr_any)
const struct in6_addr __in6addr_loopback =
{ { { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1 } } };
libc_hidden_data_def (__in6addr_loopback)
weak_alias (__in6addr_loopback, in6addr_loopback)
libc_hidden_data_weak (in6addr_loopback)
