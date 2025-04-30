/* Get descriptor for character set conversion.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1997.

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

#include <alloca.h>
#include <errno.h>
#include <iconv.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include <gconv_int.h>
#include "gconv_charset.h"


iconv_t
iconv_open (const char *tocode, const char *fromcode)
{
  __gconv_t cd;
  struct gconv_spec conv_spec;

  if (__gconv_create_spec (&conv_spec, fromcode, tocode) == NULL)
    return (iconv_t) -1;

  int res = __gconv_open (&conv_spec, &cd, 0);

  __gconv_destroy_spec (&conv_spec);

  if (__builtin_expect (res, __GCONV_OK) != __GCONV_OK)
    {
      /* We must set the error number according to the specs.  */
      if (res == __GCONV_NOCONV || res == __GCONV_NODB)
	__set_errno (EINVAL);

      cd = (iconv_t) -1;
    }

  return (iconv_t) cd;
}
