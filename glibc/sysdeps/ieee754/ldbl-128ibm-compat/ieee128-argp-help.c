/* Wrapper for argp_error and argp_failure.  IEEE128 version.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

#include <argp.h>
#include <libio/libioP.h>

void
___ieee128_argp_error (const struct argp_state *state, const char *fmt, ...)
{
  va_list ap;
  va_start (ap, fmt);
  __argp_error_internal (state, fmt, ap, PRINTF_LDBL_USES_FLOAT128);
  va_end (ap);
}
strong_alias (___ieee128_argp_error, __argp_errorieee128)

void
___ieee128_argp_failure (const struct argp_state *state, int status,
			int errnum, const char *fmt, ...)
{
  va_list ap;
  va_start (ap, fmt);
  __argp_failure_internal (state, status, errnum, fmt, ap,
			   PRINTF_LDBL_USES_FLOAT128);
  va_end (ap);
}
strong_alias (___ieee128_argp_failure, __argp_failureieee128)
