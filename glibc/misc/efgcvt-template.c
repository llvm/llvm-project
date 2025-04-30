/* Compatibility functions for floating point formatting.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/param.h>
#include <libc-lock.h>
#include <math_ldbl_opt.h>

#ifndef SPRINTF
# define SPRINTF sprintf
#endif

#define APPEND(a, b) APPEND2 (a, b)
#define APPEND2(a, b) a##b


#define FCVT_BUFFER APPEND (FUNC_PREFIX, fcvt_buffer)
#define FCVT_BUFPTR APPEND (FUNC_PREFIX, fcvt_bufptr)
#define ECVT_BUFFER APPEND (FUNC_PREFIX, ecvt_buffer)


static char FCVT_BUFFER[MAXDIG];
static char ECVT_BUFFER[MAXDIG];
libc_freeres_ptr (static char *FCVT_BUFPTR);

char *
__FCVT (FLOAT_TYPE value, int ndigit, int *decpt, int *sign)
{
  if (FCVT_BUFPTR == NULL)
    {
      if (__FCVT_R (value, ndigit, decpt, sign, FCVT_BUFFER, MAXDIG) != -1)
	return FCVT_BUFFER;

      FCVT_BUFPTR = (char *) malloc (FCVT_MAXDIG);
      if (FCVT_BUFPTR == NULL)
	return FCVT_BUFFER;
    }

  (void) __FCVT_R (value, ndigit, decpt, sign, FCVT_BUFPTR, FCVT_MAXDIG);

  return FCVT_BUFPTR;
}


char *
__ECVT (FLOAT_TYPE value, int ndigit, int *decpt, int *sign)
{
  (void) __ECVT_R (value, ndigit, decpt, sign, ECVT_BUFFER, MAXDIG);

  return ECVT_BUFFER;
}

char *
__GCVT (FLOAT_TYPE value, int ndigit, char *buf)
{
  SPRINTF (buf, "%.*" FLOAT_FMT_FLAG "g", MIN (ndigit, NDIGIT_MAX), value);
  return buf;
}
