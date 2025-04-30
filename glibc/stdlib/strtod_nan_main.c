/* Convert string for NaN payload to corresponding NaN.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

#include <ieee754.h>
#include <locale.h>
#include <math.h>
#include <stdlib.h>
#include <wchar.h>


/* If STR starts with an optional n-char-sequence as defined by ISO C
   (a sequence of ASCII letters, digits and underscores), followed by
   ENDC, return a NaN whose payload is set based on STR.  Otherwise,
   return a default NAN.  If ENDPTR is not NULL, set *ENDPTR to point
   to the character after the initial n-char-sequence.  */

FLOAT
STRTOD_NAN (const STRING_TYPE *str, STRING_TYPE **endptr, STRING_TYPE endc)
{
  const STRING_TYPE *cp = str;

  while ((*cp >= L_('0') && *cp <= L_('9'))
	 || (*cp >= L_('A') && *cp <= L_('Z'))
	 || (*cp >= L_('a') && *cp <= L_('z'))
	 || *cp == L_('_'))
    ++cp;

  FLOAT retval = NAN;
  if (*cp != endc)
    goto out;

  /* This is a system-dependent way to specify the bitmask used for
     the NaN.  We expect it to be a number which is put in the
     mantissa of the number.  */
  STRING_TYPE *endp;
  unsigned long long int mant;

  mant = STRTOULL (str, &endp, 0);
  if (endp == cp)
    SET_NAN_PAYLOAD (retval, mant);

 out:
  if (endptr != NULL)
    *endptr = (STRING_TYPE *) cp;
  return retval;
}
libc_hidden_def (STRTOD_NAN)
