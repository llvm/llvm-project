/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1996.

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

#include <ctype.h>
#include <stdlib.h>
#include <netinet/ether.h>
#include <netinet/if_ether.h>


struct ether_addr *
ether_aton_r (const char *asc, struct ether_addr *addr)
{
  size_t cnt;

  for (cnt = 0; cnt < 6; ++cnt)
    {
      unsigned int number;
      char ch;

      ch = _tolower (*asc++);
      if ((ch < '0' || ch > '9') && (ch < 'a' || ch > 'f'))
	return NULL;
      number = isdigit (ch) ? (ch - '0') : (ch - 'a' + 10);

      ch = _tolower (*asc);
      if ((cnt < 5 && ch != ':') || (cnt == 5 && ch != '\0' && !isspace (ch)))
	{
	  ++asc;
	  if ((ch < '0' || ch > '9') && (ch < 'a' || ch > 'f'))
	    return NULL;
	  number <<= 4;
	  number += isdigit (ch) ? (ch - '0') : (ch - 'a' + 10);

	  ch = *asc;
	  if (cnt < 5 && ch != ':')
	    return NULL;
	}

      /* Store result.  */
      addr->ether_addr_octet[cnt] = (unsigned char) number;

      /* Skip ':'.  */
      ++asc;
    }

  return addr;
}
libc_hidden_def (ether_aton_r)
