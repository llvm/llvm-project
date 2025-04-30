/* Copyright (C) 2011-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gmail.com>, 2011.

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

#include <uchar.h>
#include <wchar.h>


/* This is the private state used if PS is NULL.  */
static mbstate_t state;

size_t
c16rtomb (char *s, char16_t c16, mbstate_t *ps)
{
  wchar_t wc = c16;

  if (ps == NULL)
    ps = &state;

  if (s == NULL)
    {
      /* Reset any state relating to surrogate pairs.  */
      ps->__count &= 0x7fffffff;
      ps->__value.__wch = 0;
      wc = 0;
    }

  if (ps->__count & 0x80000000)
    {
      /* The previous call passed in the first surrogate of a
	 surrogate pair.  */
      ps->__count &= 0x7fffffff;
      if (wc >= 0xdc00 && wc < 0xe000)
	wc = (0x10000
	      + ((ps->__value.__wch & 0x3ff) << 10)
	      + (wc & 0x3ff));
      else
	/* This is not a low surrogate; ensure an EILSEQ error by
	   trying to decode the high surrogate as a wide character on
	   its own.  */
	wc = ps->__value.__wch;
      ps->__value.__wch = 0;
    }
  else if (wc >= 0xd800 && wc < 0xdc00)
    {
      /* The high part of a surrogate pair.  */
      ps->__count |= 0x80000000;
      ps->__value.__wch = wc;
      return 0;
    }

  return wcrtomb (s, wc, ps);
}
