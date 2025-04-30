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


/* Convert the multibyte character at S, which is no longer
   than N characters, to its `wchar_t' representation, placing
   this n *PWC and returning its length.

   Attention: this function should NEVER be intentionally used.
   The interface is completely stupid.  The state is shared between
   all conversion functions.  You should use instead the restartable
   version `mbrtowc'.  */
int
mbtowc (wchar_t *pwc, const char *s, size_t n)
{
  int result;
  static mbstate_t state;

  /* If S is NULL the function has to return null or not null
     depending on the encoding having a state depending encoding or
     not.  */
  if (s == NULL)
    {
      const struct gconv_fcts *fcts;

      /* Get the conversion functions.  */
      fcts = get_gconv_fcts (_NL_CURRENT_DATA (LC_CTYPE));

      /* This is an extension in the Unix standard which does not directly
	 violate ISO C.  */
      memset (&state, '\0', sizeof state);

      result = fcts->towc->__stateful;
    }
  else if (*s == '\0')
    {
      if (pwc != NULL)
	*pwc = L'\0';
      result = 0;
    }
  else
    {
      result = __mbrtowc (pwc, s, n, &state);

      /* The `mbrtowc' functions tell us more than we need.  Fold the -1
	 and -2 result into -1.  */
      if (result < 0)
	result = -1;
    }

  return result;
}
