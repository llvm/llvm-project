/* Convert an IPv6 scope ID from text to the internal representation.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include <net-internal.h>

#include <ctype.h>
#include <errno.h>
#include <locale.h>
#include <net/if.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/* Parse SOURCE as a scope ID for ADDRESS.  Return 0 on success and -1
   on error.  */
int
__inet6_scopeid_pton (const struct in6_addr *address, const char *scope,
                      uint32_t *result)
{
  if (IN6_IS_ADDR_LINKLOCAL (address)
      || IN6_IS_ADDR_MC_NODELOCAL (address)
      || IN6_IS_ADDR_MC_LINKLOCAL (address))
    {
      uint32_t number = __if_nametoindex (scope);
      if (number != 0)
        {
          *result = number;
          return 0;
        }
    }

  if (isdigit_l (scope[0], _nl_C_locobj_ptr))
    {
      char *end;
      unsigned long long number
        = ____strtoull_l_internal (scope, &end, /*base */ 10, /* group */ 0,
                                   _nl_C_locobj_ptr);
      if (*end == '\0' && number <= UINT32_MAX)
        {
          *result = number;
          return 0;
        }
    }

  __set_errno (EINVAL);
  return -1;
}

libc_hidden_def (__inet6_scopeid_pton)
