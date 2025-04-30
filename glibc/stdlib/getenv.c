/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include <endian.h>
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>


/* Return the value of the environment variable NAME.  This implementation
   is tuned a bit in that it assumes no environment variable has an empty
   name which of course should always be true.  We have a special case for
   one character names so that for the general case we can assume at least
   two characters which we can access.  By doing this we can avoid using the
   `strncmp' most of the time.  */
char *
getenv (const char *name)
{
  char **ep;
  uint16_t name_start;

  if (__environ == NULL || name[0] == '\0')
    return NULL;

  if (name[1] == '\0')
    {
      /* The name of the variable consists of only one character.  Therefore
	 the first two characters of the environment entry are this character
	 and a '=' character.  */
#if __BYTE_ORDER == __LITTLE_ENDIAN || !_STRING_ARCH_unaligned
      name_start = ('=' << 8) | *(const unsigned char *) name;
#else
      name_start = '=' | ((*(const unsigned char *) name) << 8);
#endif
      for (ep = __environ; *ep != NULL; ++ep)
	{
#if _STRING_ARCH_unaligned
	  uint16_t ep_start = *(uint16_t *) *ep;
#else
	  uint16_t ep_start = (((unsigned char *) *ep)[0]
			       | (((unsigned char *) *ep)[1] << 8));
#endif
	  if (name_start == ep_start)
	    return &(*ep)[2];
	}
    }
  else
    {
      size_t len = strlen (name);
#if _STRING_ARCH_unaligned
      name_start = *(const uint16_t *) name;
#else
      name_start = (((const unsigned char *) name)[0]
		    | (((const unsigned char *) name)[1] << 8));
#endif
      len -= 2;
      name += 2;

      for (ep = __environ; *ep != NULL; ++ep)
	{
#if _STRING_ARCH_unaligned
	  uint16_t ep_start = *(uint16_t *) *ep;
#else
	  uint16_t ep_start = (((unsigned char *) *ep)[0]
			       | (((unsigned char *) *ep)[1] << 8));
#endif

	  if (name_start == ep_start && !strncmp (*ep + 2, name, len)
	      && (*ep)[len + 2] == '=')
	    return &(*ep)[len + 3];
	}
    }

  return NULL;
}
libc_hidden_def (getenv)
