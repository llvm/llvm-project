/* Name server address at specified index in res_state.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#include <assert.h>
#include <resolv.h>
#include <resolv-internal.h>

struct sockaddr *
__res_get_nsaddr (res_state statp, unsigned int n)
{
  assert (n < statp->nscount);

  if (statp->nsaddr_list[n].sin_family == 0
      && statp->_u._ext.nsaddrs[n] != NULL)
    /* statp->_u._ext.nsaddrs[n] holds an address that is larger than
       struct sockaddr, and user code did not update
       statp->nsaddr_list[n].  */
    return (struct sockaddr *) statp->_u._ext.nsaddrs[n];
  else
    /* User code updated statp->nsaddr_list[n], or statp->nsaddr_list[n]
       has the same content as statp->_u._ext.nsaddrs[n].  */
    return (struct sockaddr *) (void *) &statp->nsaddr_list[n];
}
libc_hidden_def (__res_get_nsaddr)
