/* Copyright (C) 1995-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gnu.ai.mit.edu>, 1995.

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

#include <wchar.h>
#include <errno.h>


/* Parse WCS into tokens separated by characters in DELIM.  If WCS is
   NULL, the last string wcstok() was called with is used.  */
wchar_t *
wcstok (wchar_t *wcs, const wchar_t *delim, wchar_t **save_ptr)
{
  wchar_t *result;

  if (wcs == NULL)
    {
      if (*save_ptr == NULL)
	{
	  __set_errno (EINVAL);
	  return NULL;
	}
      else
	wcs = *save_ptr;
    }

  /* Scan leading delimiters.  */
  wcs += wcsspn (wcs, delim);
  if (*wcs == L'\0')
    {
      *save_ptr = NULL;
      return NULL;
    }

  /* Find the end of the token.	 */
  result = wcs;
  wcs = wcspbrk (result, delim);
  if (wcs == NULL)
    /* This token finishes the string.	*/
    *save_ptr = NULL;
  else
    {
      /* Terminate the token and make *SAVE_PTR point past it.  */
      *wcs = L'\0';
      *save_ptr = wcs + 1;
    }
  return result;
}
