/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gnu.org>, 2000.

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

#include <stdbool.h>
#include <wchar.h>
#include <wctype.h>
#include <scratch_buffer.h>

#include "../locale/outdigits.h"
#include "../locale/outdigitswc.h"

static CHAR_T *
_i18n_number_rewrite (CHAR_T *w, CHAR_T *rear_ptr, CHAR_T *end)
{
#ifdef COMPILE_WPRINTF
# define decimal NULL
# define thousands NULL
#else
  char decimal[MB_LEN_MAX + 1];
  char thousands[MB_LEN_MAX + 1];
#endif

  /* "to_outpunct" is a map from ASCII decimal point and thousands-sep
     to their equivalent in locale. This is defined for locales which
     use extra decimal point and thousands-sep.  */
  wctrans_t map = __wctrans ("to_outpunct");
  wint_t wdecimal = __towctrans (L'.', map);
  wint_t wthousands = __towctrans (L',', map);

#ifndef COMPILE_WPRINTF
  if (__glibc_unlikely (map != NULL))
    {
      mbstate_t state;
      memset (&state, '\0', sizeof (state));

      size_t n = __wcrtomb (decimal, wdecimal, &state);
      if (n == (size_t) -1)
	memcpy (decimal, ".", 2);
      else
	decimal[n] = '\0';

      memset (&state, '\0', sizeof (state));

      n = __wcrtomb (thousands, wthousands, &state);
      if (n == (size_t) -1)
	memcpy (thousands, ",", 2);
      else
	thousands[n] = '\0';
    }
#endif

  /* Copy existing string so that nothing gets overwritten.  */
  CHAR_T *src;
  struct scratch_buffer buffer;
  scratch_buffer_init (&buffer);
  if (!scratch_buffer_set_array_size (&buffer, rear_ptr - w, sizeof (CHAR_T)))
    /* If we cannot allocate the memory don't rewrite the string.
       It is better than nothing.  */
    return w;
  src = buffer.data;

  CHAR_T *s = (CHAR_T *) __mempcpy (src, w,
				    (rear_ptr - w) * sizeof (CHAR_T));

  w = end;

  /* Process all characters in the string.  */
  while (--s >= src)
    {
      if (*s >= '0' && *s <= '9')
	{
	  if (sizeof (CHAR_T) == 1)
	    w = (CHAR_T *) outdigit_value ((char *) w, *s - '0');
	  else
	    *--w = (CHAR_T) outdigitwc_value (*s - '0');
	}
      else if (__builtin_expect (map == NULL, 1) || (*s != '.' && *s != ','))
	*--w = *s;
      else
	{
	  if (sizeof (CHAR_T) == 1)
	    {
	      const char *outpunct = *s == '.' ? decimal : thousands;
	      size_t dlen = strlen (outpunct);

	      w -= dlen;
	      while (dlen-- > 0)
		w[dlen] = outpunct[dlen];
	    }
	  else
	    *--w = *s == '.' ? (CHAR_T) wdecimal : (CHAR_T) wthousands;
	}
    }

  scratch_buffer_free (&buffer);
  return w;
}
