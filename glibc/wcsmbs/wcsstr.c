/* Copyright (C) 1995-2021 Free Software Foundation, Inc.
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

/*
 * The original strstr() file contains the following comment:
 *
 * My personal strstr() implementation that beats most other algorithms.
 * Until someone tells me otherwise, I assume that this is the
 * fastest implementation of strstr() in C.
 * I deliberately chose not to comment it.  You should have at least
 * as much fun trying to understand it, as I had to write it :-).
 *
 * Stephen R. van den Berg, berg@pool.informatik.rwth-aachen.de */

#include <wchar.h>

wchar_t *
wcsstr (const wchar_t *haystack, const wchar_t *needle)
{
  wchar_t b, c;

  if ((b = *needle) != L'\0')
    {
      haystack--;				/* possible ANSI violation */
      do
	if ((c = *++haystack) == L'\0')
	  goto ret0;
      while (c != b);

      if (!(c = *++needle))
	goto foundneedle;
      ++needle;
      goto jin;

      for (;;)
	{
	  wchar_t a;
	  const wchar_t *rhaystack, *rneedle;

	  do
	    {
	      if (!(a = *++haystack))
		goto ret0;
	      if (a == b)
		break;
	      if ((a = *++haystack) == L'\0')
		goto ret0;
shloop:	      ;
	    }
	  while (a != b);

jin:	  if (!(a = *++haystack))
	    goto ret0;

	  if (a != c)
	    goto shloop;

	  if (*(rhaystack = haystack-- + 1) == (a = *(rneedle = needle)))
	    do
	      {
		if (a == L'\0')
		  goto foundneedle;
		if (*++rhaystack != (a = *++needle))
		  break;
		if (a == L'\0')
		  goto foundneedle;
	      }
	    while (*++rhaystack == (a = *++needle));

	  needle = rneedle;		  /* took the register-poor approach */

	  if (a == L'\0')
	    break;
	}
    }
foundneedle:
  return (wchar_t*) haystack;
ret0:
  return NULL;
}
/* This alias is for backward compatibility with drafts of the ISO C
   standard.  Unfortunately the Unix(TM) standard requires this name.  */
weak_alias (wcsstr, wcswcs)
