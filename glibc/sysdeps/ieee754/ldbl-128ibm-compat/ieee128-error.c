/* Wrappers for error.h functions.  IEEE128 version.
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

#include <error.h>
#include <stdarg.h>
#include <libio/libioP.h>

#define IEEE128_ALIAS(name) \
  strong_alias (___ieee128_##name, __##name##ieee128)

#define IEEE128_DECL(name) ___ieee128_##name

void
IEEE128_DECL (error) (int status, int errnum, const char *message, ...)
{
  va_list ap;
  va_start (ap, message);
  __error_internal (status, errnum, message, ap,
		    PRINTF_LDBL_USES_FLOAT128);
  va_end (ap);
}
IEEE128_ALIAS (error)

void
IEEE128_DECL (error_at_line) (int status, int errnum,
			      const char *file_name,
			      unsigned int line_number,
			      const char *message, ...)
{
  va_list ap;
  va_start (ap, message);
  __error_at_line_internal (status, errnum, file_name, line_number,
			    message, ap, PRINTF_LDBL_USES_FLOAT128);
  va_end (ap);
}
IEEE128_ALIAS (error_at_line)
