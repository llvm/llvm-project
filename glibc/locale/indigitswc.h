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

#include <assert.h>
#include <langinfo.h>

/* Look up the value of the next multibyte character and return its numerical
   value if it is one of the digits known in the locale.  If *DECIDED is
   -1 this means it is not yet decided which form it is and we have to
   search through all available digits.  Otherwise we know which script
   the digits are from.  */
static inline int
indigitwc_value (wchar_t wc, int *decided)
{
  int from_level;
  int to_level;
  const wchar_t *wcdigits[10];
  int n;

  if (*decided != -1)
    from_level = to_level = *decided;
  else
    {
      from_level = 0;
      to_level = _NL_CURRENT_WORD (LC_CTYPE, _NL_CTYPE_INDIGITS_WC_LEN) - 1;
      assert (from_level <= to_level);
    }

  /* In this round we get the pointer to the digit strings and also perform
     the first round of comparisons.  */
  for (n = 0; n < 10; ++n)
    {
      /* Get the string for the digits with value N.  */
      wcdigits[n] = _NL_CURRENT (LC_CTYPE, _NL_CTYPE_INDIGITS0_WC + n);
      wcdigits[n] += from_level;

      if (wc == *wcdigits[n])
	{
	  /* Found it.  */
	  if (*decided == -1)
	    *decided = 0;
	  return n;
	}

      /* Advance the pointer to the next string.  */
      ++wcdigits[n];
    }

  /* Now perform the remaining tests.  */
  while (++from_level <= to_level)
    {
      /* Search all ten digits of this level.  */
      for (n = 0; n < 10; ++n)
	{
	  if (wc == *wcdigits[n])
	    {
	      /* Found it.  */
	      if (*decided == -1)
		*decided = from_level;
	      return n;
	    }

	  /* Advance the pointer to the next string.  */
	  ++wcdigits[n];
	}
    }

  /* If we reach this point no matching digit was found.  */
  return -1;
}
