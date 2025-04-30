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

#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <gconv.h>
#include <wcsmbs/wcsmbsload.h>


/* Internal state.  */
static mbstate_t state;

/* Return the length of the multibyte character (if there is one)
   at S which is no longer than N characters.
   The ISO C standard says that the `mblen' function must not change
   the state of the `mbtowc' function.  */
int
mblen (const char *s, size_t n)
{
  int result;

  /* If S is NULL the function has to return null or not null
     depending on the encoding having a state depending encoding or
     not.  */
  if (s == NULL)
    {
      const struct gconv_fcts *fcts;

      /* Get the conversion functions.  */
      fcts = get_gconv_fcts (_NL_CURRENT_DATA (LC_CTYPE));

      /* Reset the state.  */
      memset (&state, '\0', sizeof state);

      result = fcts->towc->__stateful;
    }
  else if (*s == '\0')
    /* According to the ISO C 89 standard this is the expected behaviour.  */
    result = 0;
  else
    {
      memset (&state, '\0', sizeof state);

      result = __mbrtowc (NULL, s, n, &state);

      /* The `mbrtowc' functions tell us more than we need.  Fold the -1
	 and -2 result into -1.  */
      if (result < 0)
	result = -1;
    }

  return result;
}
