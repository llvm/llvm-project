/* Wrappers for err.h functions.  IEEE128 version.
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

#include <err.h>
#include <stdarg.h>
#include <libio/libioP.h>

#define VA(call)							\
{									\
  va_list ap;								\
  va_start (ap, format);						\
  IEEE128_CALL (call);							\
  va_end (ap);								\
}

#define IEEE128_ALIAS(name) \
  strong_alias (___ieee128_##name, __##name##ieee128)

#define IEEE128_DECL(name) ___ieee128_##name
#define IEEE128_CALL(name) ___ieee128_##name

void
IEEE128_DECL (vwarn) (const char *format, __gnuc_va_list ap)
{
  __vwarn_internal (format, ap, PRINTF_LDBL_USES_FLOAT128);
}
IEEE128_ALIAS (vwarn)

void
IEEE128_DECL (vwarnx) (const char *format, __gnuc_va_list ap)
{
  __vwarnx_internal (format, ap, PRINTF_LDBL_USES_FLOAT128);
}
IEEE128_ALIAS (vwarnx)

void
IEEE128_DECL (warn) (const char *format, ...)
{
  VA (vwarn (format, ap))
}
IEEE128_ALIAS (warn)

void
IEEE128_DECL (warnx) (const char *format, ...)
{
  VA (vwarnx (format, ap))
}
IEEE128_ALIAS (warnx)

void
IEEE128_DECL (verr) (int status, const char *format, __gnuc_va_list ap)
{
  IEEE128_CALL (vwarn) (format, ap);
  exit (status);
}
IEEE128_ALIAS (verr)

void
IEEE128_DECL (verrx) (int status, const char *format, __gnuc_va_list ap)
{
  IEEE128_CALL (vwarnx) (format, ap);
  exit (status);
}
IEEE128_ALIAS (verrx)

void
IEEE128_DECL (err) (int status, const char *format, ...)
{
  VA (verr (status, format, ap))
}
IEEE128_ALIAS (err)

void
IEEE128_DECL (errx) (int status, const char *format, ...)
{
  VA (verrx (status, format, ap))
}
IEEE128_ALIAS (errx)

hidden_def (___ieee128_warn)
hidden_def (___ieee128_warnx)
hidden_def (___ieee128_vwarn)
hidden_def (___ieee128_vwarnx)
hidden_def (___ieee128_verr)
hidden_def (___ieee128_verrx)
