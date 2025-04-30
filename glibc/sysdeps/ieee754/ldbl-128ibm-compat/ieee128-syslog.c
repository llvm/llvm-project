/* Wrapper for syslog.  IEEE128 version.
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

#include <stdarg.h>
#include <libioP.h>
#include <syslog.h>

void
___ieee128_syslog (int pri, const char *fmt, ...)
{
  va_list ap;

  va_start (ap, fmt);
  __vsyslog_internal (pri, fmt, ap, PRINTF_LDBL_USES_FLOAT128);
  va_end (ap);
}
strong_alias (___ieee128_syslog, __syslogieee128)
hidden_def (___ieee128_syslog)

void
___ieee128_vsyslog (int pri, const char *fmt, va_list ap)
{
  __vsyslog_internal (pri, fmt, ap, PRINTF_LDBL_USES_FLOAT128);
}
strong_alias (___ieee128_vsyslog, __vsyslogieee128)

void
___ieee128_syslog_chk (int pri, int flag, const char *fmt, ...)
{
  va_list ap;

  unsigned int mode = PRINTF_LDBL_USES_FLOAT128;
  if (flag > 0)
    mode |= PRINTF_FORTIFY;

  va_start (ap, fmt);
  __vsyslog_internal (pri, fmt, ap, mode);
  va_end (ap);
}
strong_alias (___ieee128_syslog_chk, __syslog_chkieee128)

void
___ieee128_vsyslog_chk (int pri, int flag, const char *fmt, va_list ap)
{
  unsigned int mode = PRINTF_LDBL_USES_FLOAT128;
  if (flag > 0)
    mode |= PRINTF_FORTIFY;

  __vsyslog_internal (pri, fmt, ap, mode);
}
strong_alias (___ieee128_vsyslog_chk, __vsyslog_chkieee128)
