/* Charset name normalization.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 2001.

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
#include <locale.h>
#include <stdbool.h>
#include <string.h>
#include <sys/stat.h>
#include <stdlib.h>
#include "gconv_int.h"


/* An iconv encoding is in the form of a triplet, with parts separated by
   a '/' character.  The first part is the standard name, the second part is
   the character set, and the third part is the error handler.  If the first
   part is sufficient to identify both the standard and the character set
   then the second part can be empty e.g. UTF-8//.  If the first part is not
   sufficient to identify both the standard and the character set then the
   second part is required e.g. ISO-10646/UTF8/.  If neither the first or
   second parts are provided e.g. //, then the current locale is used.
   The actual values used in the first and second parts are not entirely
   relevant to the implementation.  The values themselves are used in a hash
   table to lookup modules and so the naming convention of the first two parts
   is somewhat arbitrary and only helps locate the entries in the cache.
   The third part is the error handler and is comprised of a ',' or '/'
   separated list of suffixes.  Currently, we support "TRANSLIT" for
   transliteration and "IGNORE" for ignoring conversion errors due to
   unrecognized input characters.  */
#define GCONV_TRIPLE_SEPARATOR "/"
#define GCONV_SUFFIX_SEPARATOR ","
#define GCONV_TRANSLIT_SUFFIX "TRANSLIT"
#define GCONV_IGNORE_ERRORS_SUFFIX "IGNORE"


/* This function copies in-order, characters from the source 's' that are
   either alpha-numeric or one in one of these: "_-.,:/" - into the destination
   'wp' while dropping all other characters.  In the process, it converts all
   alphabetical characters to upper case.  It then appends up to two '/'
   characters so that the total number of '/'es in the destination is 2.  */
static inline void __attribute__ ((unused, always_inline))
strip (char *wp, const char *s)
{
  int slash_count = 0;

  while (*s != '\0')
    {
      if (__isalnum_l (*s, _nl_C_locobj_ptr)
	  || *s == '_' || *s == '-' || *s == '.' || *s == ',' || *s == ':')
	*wp++ = __toupper_l (*s, _nl_C_locobj_ptr);
      else if (*s == '/')
	{
	  if (++slash_count == 3)
	    break;
	  *wp++ = '/';
	}
      ++s;
    }

  while (slash_count++ < 2)
    *wp++ = '/';

  *wp = '\0';
}


static inline char * __attribute__ ((unused, always_inline))
upstr (char *dst, const char *str)
{
  char *cp = dst;
  while ((*cp++ = __toupper_l (*str++, _nl_C_locobj_ptr)) != '\0')
    /* nothing */;
  return dst;
}
