/* *printf* family compatibility routines for IEEE double as long double
   Copyright (C) 2006-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@cygnus.com>, 2006.

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

/* This file may define some of the deprecated scanf variants.  */
#include <features.h>
#undef __GLIBC_USE_DEPRECATED_SCANF
#define __GLIBC_USE_DEPRECATED_SCANF 1

#include <argp.h>
#include <err.h>
#include <error.h>
#include <stdarg.h>
#include <stdio.h>
#include <libio/strfile.h>
#include <math.h>
#include <wchar.h>
#include <printf.h>
#include <monetary.h>
#include <locale/localeinfo.h>
#include <sys/syslog.h>
#include <libc-lock.h>

#include "nldbl-compat.h"

libc_hidden_proto (__nldbl_vsscanf)
libc_hidden_proto (__nldbl_vfscanf)
libc_hidden_proto (__nldbl_vfwscanf)
libc_hidden_proto (__nldbl_vswscanf)
libc_hidden_proto (__nldbl___isoc99_vsscanf)
libc_hidden_proto (__nldbl___isoc99_vfscanf)
libc_hidden_proto (__nldbl___isoc99_vswscanf)
libc_hidden_proto (__nldbl___isoc99_vfwscanf)

/* Compatibility with IEEE double as long double.
   IEEE quad long double is used by default for most programs, so
   we don't need to split this into one file per function for the
   sake of statically linked programs.  */

int
attribute_compat_text_section
__nldbl___asprintf (char **string_ptr, const char *fmt, ...)
{
  va_list ap;
  int ret;

  va_start (ap, fmt);
  ret = __vasprintf_internal (string_ptr, fmt, ap, PRINTF_LDBL_IS_DBL);
  va_end (ap);

  return ret;
}
weak_alias (__nldbl___asprintf, __nldbl_asprintf)

int
attribute_compat_text_section
__nldbl_dprintf (int d, const char *fmt, ...)
{
  va_list ap;
  int ret;

  va_start (ap, fmt);
  ret = __vdprintf_internal (d, fmt, ap, PRINTF_LDBL_IS_DBL);
  va_end (ap);

  return ret;
}

int
attribute_compat_text_section
__nldbl_fprintf (FILE *stream, const char *fmt, ...)
{
  va_list ap;
  int ret;

  va_start (ap, fmt);
  ret = __vfprintf_internal (stream, fmt, ap, PRINTF_LDBL_IS_DBL);
  va_end (ap);

  return ret;
}
weak_alias (__nldbl_fprintf, __nldbl__IO_fprintf)

int
attribute_compat_text_section weak_function
__nldbl_fwprintf (FILE *stream, const wchar_t *fmt, ...)
{
  va_list ap;
  int ret;

  va_start (ap, fmt);
  ret = __vfwprintf_internal (stream, fmt, ap, PRINTF_LDBL_IS_DBL);
  va_end (ap);

  return ret;
}

int
attribute_compat_text_section
__nldbl_printf (const char *fmt, ...)
{
  va_list ap;
  int ret;

  va_start (ap, fmt);
  ret = __vfprintf_internal (stdout, fmt, ap, PRINTF_LDBL_IS_DBL);
  va_end (ap);

  return ret;
}
strong_alias (__nldbl_printf, __nldbl__IO_printf)

int
attribute_compat_text_section
__nldbl_sprintf (char *s, const char *fmt, ...)
{
  va_list ap;
  int ret;

  va_start (ap, fmt);
  ret = __vsprintf_internal (s, -1, fmt, ap, PRINTF_LDBL_IS_DBL);
  va_end (ap);

  return ret;
}
strong_alias (__nldbl_sprintf, __nldbl__IO_sprintf)

int
attribute_compat_text_section
__nldbl_vfprintf (FILE *s, const char *fmt, va_list ap)
{
  return __vfprintf_internal (s, fmt, ap, PRINTF_LDBL_IS_DBL);
}
strong_alias (__nldbl_vfprintf, __nldbl__IO_vfprintf)

int
attribute_compat_text_section
__nldbl___vsprintf (char *string, const char *fmt, va_list ap)
{
  return __vsprintf_internal (string, -1, fmt, ap, PRINTF_LDBL_IS_DBL);
}
strong_alias (__nldbl___vsprintf, __nldbl__IO_vsprintf)
weak_alias (__nldbl___vsprintf, __nldbl_vsprintf)

int
attribute_compat_text_section
__nldbl_obstack_vprintf (struct obstack *obstack, const char *fmt,
			 va_list ap)
{
  return __obstack_vprintf_internal (obstack, fmt, ap, PRINTF_LDBL_IS_DBL);
}

int
attribute_compat_text_section
__nldbl_obstack_printf (struct obstack *obstack, const char *fmt, ...)
{
  int ret;
  va_list ap;
  va_start (ap, fmt);
  ret = __obstack_vprintf_internal (obstack, fmt, ap, PRINTF_LDBL_IS_DBL);
  va_end (ap);
  return ret;
}

int
attribute_compat_text_section weak_function
__nldbl_snprintf (char *s, size_t maxlen, const char *fmt, ...)
{
  va_list ap;
  int ret;

  va_start (ap, fmt);
  ret = __vsnprintf_internal (s, maxlen, fmt, ap, PRINTF_LDBL_IS_DBL);
  va_end (ap);

  return ret;
}

int
attribute_compat_text_section
__nldbl_swprintf (wchar_t *s, size_t n, const wchar_t *fmt, ...)
{
  va_list ap;
  int ret;

  va_start (ap, fmt);
  ret = __vswprintf_internal (s, n, fmt, ap, PRINTF_LDBL_IS_DBL);
  va_end (ap);

  return ret;
}

int
attribute_compat_text_section weak_function
__nldbl_vasprintf (char **result_ptr, const char *fmt, va_list ap)
{
  return __vasprintf_internal (result_ptr, fmt, ap, PRINTF_LDBL_IS_DBL);
}

int
attribute_compat_text_section
__nldbl_vdprintf (int d, const char *fmt, va_list ap)
{
  return __vdprintf_internal (d, fmt, ap, PRINTF_LDBL_IS_DBL);
}

int
attribute_compat_text_section weak_function
__nldbl_vfwprintf (FILE *s, const wchar_t *fmt, va_list ap)
{
  return __vfwprintf_internal (s, fmt, ap, PRINTF_LDBL_IS_DBL);
}

int
attribute_compat_text_section
__nldbl_vprintf (const char *fmt, va_list ap)
{
  return __vfprintf_internal (stdout, fmt, ap, PRINTF_LDBL_IS_DBL);
}

int
attribute_compat_text_section
__nldbl_vsnprintf (char *string, size_t maxlen, const char *fmt,
		   va_list ap)
{
  return __vsnprintf_internal (string, maxlen, fmt, ap, PRINTF_LDBL_IS_DBL);
}
weak_alias (__nldbl_vsnprintf, __nldbl___vsnprintf)

int
attribute_compat_text_section weak_function
__nldbl_vswprintf (wchar_t *string, size_t maxlen, const wchar_t *fmt,
		   va_list ap)
{
  return __vswprintf_internal (string, maxlen, fmt, ap, PRINTF_LDBL_IS_DBL);
}

int
attribute_compat_text_section
__nldbl_vwprintf (const wchar_t *fmt, va_list ap)
{
  return __vfwprintf_internal (stdout, fmt, ap, PRINTF_LDBL_IS_DBL);
}

int
attribute_compat_text_section
__nldbl_wprintf (const wchar_t *fmt, ...)
{
  va_list ap;
  int ret;

  va_start (ap, fmt);
  ret = __vfwprintf_internal (stdout, fmt, ap, PRINTF_LDBL_IS_DBL);
  va_end (ap);

  return ret;
}

#if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_29)
int
attribute_compat_text_section
__nldbl__IO_vfscanf (FILE *s, const char *fmt, va_list ap, int *errp)
{
  int ret = __vfscanf_internal (s, fmt, ap, SCANF_LDBL_IS_DBL);
  if (__glibc_unlikely (errp != 0))
    *errp = (ret == -1);
  return ret;
}
#endif

int
attribute_compat_text_section
__nldbl___vfscanf (FILE *s, const char *fmt, va_list ap)
{
  return __vfscanf_internal (s, fmt, ap, SCANF_LDBL_IS_DBL);
}
weak_alias (__nldbl___vfscanf, __nldbl_vfscanf)
libc_hidden_def (__nldbl_vfscanf)

int
attribute_compat_text_section
__nldbl_sscanf (const char *s, const char *fmt, ...)
{
  _IO_strfile sf;
  FILE *f = _IO_strfile_read (&sf, s);
  va_list ap;
  int ret;

  va_start (ap, fmt);
  ret = __vfscanf_internal (f, fmt, ap, SCANF_LDBL_IS_DBL);
  va_end (ap);

  return ret;
}
strong_alias (__nldbl_sscanf, __nldbl__IO_sscanf)

int
attribute_compat_text_section
__nldbl___vsscanf (const char *s, const char *fmt, va_list ap)
{
  _IO_strfile sf;
  FILE *f = _IO_strfile_read (&sf, s);
  return __vfscanf_internal (f, fmt, ap, SCANF_LDBL_IS_DBL);
}
weak_alias (__nldbl___vsscanf, __nldbl_vsscanf)
libc_hidden_def (__nldbl_vsscanf)

int
attribute_compat_text_section weak_function
__nldbl_vscanf (const char *fmt, va_list ap)
{
  return __vfscanf_internal (stdin, fmt, ap, SCANF_LDBL_IS_DBL);
}

int
attribute_compat_text_section
__nldbl_fscanf (FILE *stream, const char *fmt, ...)
{
  va_list ap;
  int ret;

  va_start (ap, fmt);
  ret = __vfscanf_internal (stream, fmt, ap, SCANF_LDBL_IS_DBL);
  va_end (ap);

  return ret;
}

int
attribute_compat_text_section
__nldbl_scanf (const char *fmt, ...)
{
  va_list ap;
  int ret;

  va_start (ap, fmt);
  ret = __vfscanf_internal (stdin, fmt, ap, SCANF_LDBL_IS_DBL);
  va_end (ap);

  return ret;
}

int
attribute_compat_text_section
__nldbl_vfwscanf (FILE *s, const wchar_t *fmt, va_list ap)
{
  return __vfwscanf_internal (s, fmt, ap, SCANF_LDBL_IS_DBL);
}
libc_hidden_def (__nldbl_vfwscanf)

int
attribute_compat_text_section
__nldbl_swscanf (const wchar_t *s, const wchar_t *fmt, ...)
{
  _IO_strfile sf;
  struct _IO_wide_data wd;
  FILE *f = _IO_strfile_readw (&sf, &wd, s);
  va_list ap;
  int ret;

  va_start (ap, fmt);
  ret = __vfwscanf_internal (f, fmt, ap, SCANF_LDBL_IS_DBL);
  va_end (ap);

  return ret;
}

int
attribute_compat_text_section
__nldbl_vswscanf (const wchar_t *s, const wchar_t *fmt, va_list ap)
{
  _IO_strfile sf;
  struct _IO_wide_data wd;
  FILE *f = _IO_strfile_readw (&sf, &wd, s);

  return __vfwscanf_internal (f, fmt, ap, SCANF_LDBL_IS_DBL);
}
libc_hidden_def (__nldbl_vswscanf)

int
attribute_compat_text_section weak_function
__nldbl_vwscanf (const wchar_t *fmt, va_list ap)
{
  return __vfwscanf_internal (stdin, fmt, ap, SCANF_LDBL_IS_DBL);
}

int
attribute_compat_text_section
__nldbl_fwscanf (FILE *stream, const wchar_t *fmt, ...)
{
  va_list ap;
  int ret;

  va_start (ap, fmt);
  ret = __vfwscanf_internal (stream, fmt, ap, SCANF_LDBL_IS_DBL);
  va_end (ap);

  return ret;
}

int
attribute_compat_text_section
__nldbl_wscanf (const wchar_t *fmt, ...)
{
  va_list ap;
  int ret;

  va_start (ap, fmt);
  ret = __vfwscanf_internal (stdin, fmt, ap, SCANF_LDBL_IS_DBL);
  va_end (ap);

  return ret;
}

int
attribute_compat_text_section
__nldbl___fprintf_chk (FILE *stream, int flag, const char *fmt, ...)
{
  va_list ap;
  int ret;
  unsigned int mode = PRINTF_LDBL_IS_DBL;
  if (flag > 0)
    mode |= PRINTF_FORTIFY;

  va_start (ap, fmt);
  ret = __vfprintf_internal (stream, fmt, ap, mode);
  va_end (ap);

  return ret;
}

int
attribute_compat_text_section
__nldbl___fwprintf_chk (FILE *stream, int flag, const wchar_t *fmt, ...)
{
  va_list ap;
  int ret;
  unsigned int mode = PRINTF_LDBL_IS_DBL;
  if (flag > 0)
    mode |= PRINTF_FORTIFY;

  va_start (ap, fmt);
  ret = __vfwprintf_internal (stream, fmt, ap, mode);
  va_end (ap);

  return ret;
}

int
attribute_compat_text_section
__nldbl___printf_chk (int flag, const char *fmt, ...)
{
  va_list ap;
  int ret;
  unsigned int mode = PRINTF_LDBL_IS_DBL;
  if (flag > 0)
    mode |= PRINTF_FORTIFY;

  va_start (ap, fmt);
  ret = __vfprintf_internal (stdout, fmt, ap, mode);
  va_end (ap);

  return ret;
}

int
attribute_compat_text_section
__nldbl___snprintf_chk (char *s, size_t maxlen, int flag, size_t slen,
			const char *fmt, ...)
{
  if (__glibc_unlikely (slen < maxlen))
    __chk_fail ();

  va_list ap;
  int ret;
  unsigned int mode = PRINTF_LDBL_IS_DBL;
  if (flag > 0)
    mode |= PRINTF_FORTIFY;

  va_start (ap, fmt);
  ret = __vsnprintf_internal (s, maxlen, fmt, ap, mode);
  va_end (ap);

  return ret;
}

int
attribute_compat_text_section
__nldbl___sprintf_chk (char *s, int flag, size_t slen, const char *fmt, ...)
{
  if (slen == 0)
    __chk_fail ();

  va_list ap;
  int ret;
  unsigned int mode = PRINTF_LDBL_IS_DBL;
  if (flag > 0)
    mode |= PRINTF_FORTIFY;

  va_start (ap, fmt);
  ret = __vsprintf_internal (s, slen, fmt, ap, mode);
  va_end (ap);

  return ret;
}

int
attribute_compat_text_section
__nldbl___swprintf_chk (wchar_t *s, size_t maxlen, int flag, size_t slen,
			const wchar_t *fmt, ...)
{
  if (__glibc_unlikely (slen < maxlen))
    __chk_fail ();

  va_list ap;
  int ret;
  unsigned int mode = PRINTF_LDBL_IS_DBL;
  if (flag > 0)
    mode |= PRINTF_FORTIFY;

  va_start (ap, fmt);
  ret = __vswprintf_internal (s, maxlen, fmt, ap, mode);
  va_end (ap);

  return ret;
}

int
attribute_compat_text_section
__nldbl___vfprintf_chk (FILE *s, int flag, const char *fmt, va_list ap)
{
  unsigned int mode = PRINTF_LDBL_IS_DBL;
  if (flag > 0)
    mode |= PRINTF_FORTIFY;

  return __vfprintf_internal (s, fmt, ap, mode);
}

int
attribute_compat_text_section
__nldbl___vfwprintf_chk (FILE *s, int flag, const wchar_t *fmt, va_list ap)
{
  unsigned int mode = PRINTF_LDBL_IS_DBL;
  if (flag > 0)
    mode |= PRINTF_FORTIFY;

  return __vfwprintf_internal (s, fmt, ap, mode);
}

int
attribute_compat_text_section
__nldbl___vprintf_chk (int flag, const char *fmt, va_list ap)
{
  unsigned int mode = PRINTF_LDBL_IS_DBL;
  if (flag > 0)
    mode |= PRINTF_FORTIFY;

  return __vfprintf_internal (stdout, fmt, ap, mode);
}

int
attribute_compat_text_section
__nldbl___vsnprintf_chk (char *string, size_t maxlen, int flag, size_t slen,
			 const char *fmt, va_list ap)
{
  if (__glibc_unlikely (slen < maxlen))
    __chk_fail ();

  unsigned int mode = PRINTF_LDBL_IS_DBL;
  if (flag > 0)
    mode |= PRINTF_FORTIFY;

  return __vsnprintf_internal (string, maxlen, fmt, ap, mode);
}

int
attribute_compat_text_section
__nldbl___vsprintf_chk (char *string, int flag, size_t slen, const char *fmt,
			va_list ap)
{
  if (slen == 0)
    __chk_fail ();

  unsigned int mode = PRINTF_LDBL_IS_DBL;
  if (flag > 0)
    mode |= PRINTF_FORTIFY;

  return __vsprintf_internal (string, slen, fmt, ap, mode);
}

int
attribute_compat_text_section
__nldbl___vswprintf_chk (wchar_t *string, size_t maxlen, int flag, size_t slen,
			 const wchar_t *fmt, va_list ap)
{
  if (__glibc_unlikely (slen < maxlen))
    __chk_fail ();

  unsigned int mode = PRINTF_LDBL_IS_DBL;
  if (flag > 0)
    mode |= PRINTF_FORTIFY;

  return __vswprintf_internal (string, maxlen, fmt, ap, mode);
}

int
attribute_compat_text_section
__nldbl___vwprintf_chk (int flag, const wchar_t *fmt, va_list ap)
{
  unsigned int mode = PRINTF_LDBL_IS_DBL;
  if (flag > 0)
    mode |= PRINTF_FORTIFY;

  return __vfwprintf_internal (stdout, fmt, ap, mode);
}

int
attribute_compat_text_section
__nldbl___wprintf_chk (int flag, const wchar_t *fmt, ...)
{
  va_list ap;
  int ret;
  unsigned int mode = PRINTF_LDBL_IS_DBL;
  if (flag > 0)
    mode |= PRINTF_FORTIFY;

  va_start (ap, fmt);
  ret = __vfwprintf_internal (stdout, fmt, ap, mode);
  va_end (ap);

  return ret;
}

int
attribute_compat_text_section
__nldbl___vasprintf_chk (char **ptr, int flag, const char *fmt, va_list ap)
{
  unsigned int mode = PRINTF_LDBL_IS_DBL;
  if (flag > 0)
    mode |= PRINTF_FORTIFY;

  return __vasprintf_internal (ptr, fmt, ap, mode);
}

int
attribute_compat_text_section
__nldbl___asprintf_chk (char **ptr, int flag, const char *fmt, ...)
{
  va_list ap;
  int ret;
  unsigned int mode = PRINTF_LDBL_IS_DBL;
  if (flag > 0)
    mode |= PRINTF_FORTIFY;

  va_start (ap, fmt);
  ret = __vasprintf_internal (ptr, fmt, ap, mode);
  va_end (ap);

  return ret;
}

int
attribute_compat_text_section
__nldbl___vdprintf_chk (int d, int flag, const char *fmt, va_list ap)
{
  unsigned int mode = PRINTF_LDBL_IS_DBL;
  if (flag > 0)
    mode |= PRINTF_FORTIFY;

  return __vdprintf_internal (d, fmt, ap, mode);
}

int
attribute_compat_text_section
__nldbl___dprintf_chk (int d, int flag, const char *fmt, ...)
{
  va_list ap;
  int ret;
  unsigned int mode = PRINTF_LDBL_IS_DBL;
  if (flag > 0)
    mode |= PRINTF_FORTIFY;

  va_start (ap, fmt);
  ret = __vdprintf_internal (d, fmt, ap, mode);
  va_end (ap);

  return ret;
}

int
attribute_compat_text_section
__nldbl___obstack_vprintf_chk (struct obstack *obstack, int flag,
			       const char *fmt, va_list ap)
{
  unsigned int mode = PRINTF_LDBL_IS_DBL;
  if (flag > 0)
    mode |= PRINTF_FORTIFY;

  return __obstack_vprintf_internal (obstack, fmt, ap, mode);
}

int
attribute_compat_text_section
__nldbl___obstack_printf_chk (struct obstack *obstack, int flag,
			      const char *fmt, ...)
{
  va_list ap;
  int ret;
  unsigned int mode = PRINTF_LDBL_IS_DBL;
  if (flag > 0)
    mode |= PRINTF_FORTIFY;

  va_start (ap, fmt);
  ret = __obstack_vprintf_internal (obstack, fmt, ap, mode);
  va_end (ap);

  return ret;
}

extern __typeof (printf_size) __printf_size;

int
attribute_compat_text_section
__nldbl_printf_size (FILE *fp, const struct printf_info *info,
		     const void *const *args)
{
  struct printf_info info_no_ldbl = *info;

  info_no_ldbl.is_long_double = 0;
  return __printf_size (fp, &info_no_ldbl, args);
}

extern __typeof (__printf_fp) ___printf_fp;

int
attribute_compat_text_section
__nldbl___printf_fp (FILE *fp, const struct printf_info *info,
		     const void *const *args)
{
  struct printf_info info_no_ldbl = *info;

  info_no_ldbl.is_long_double = 0;
  return ___printf_fp (fp, &info_no_ldbl, args);
}

ssize_t
attribute_compat_text_section
__nldbl_strfmon (char *s, size_t maxsize, const char *format, ...)
{
  va_list ap;
  ssize_t ret;

  va_start (ap, format);
  ret = __vstrfmon_l_internal (s, maxsize, _NL_CURRENT_LOCALE, format, ap,
			       STRFMON_LDBL_IS_DBL);
  va_end (ap);
  return ret;
}

ssize_t
attribute_compat_text_section
__nldbl___strfmon_l (char *s, size_t maxsize, locale_t loc,
		     const char *format, ...)
{
  va_list ap;
  ssize_t ret;

  va_start (ap, format);
  ret = __vstrfmon_l_internal (s, maxsize, loc, format, ap,
			       STRFMON_LDBL_IS_DBL);
  va_end (ap);
  return ret;
}
weak_alias (__nldbl___strfmon_l, __nldbl_strfmon_l)

ssize_t
attribute_compat_text_section
__nldbl___vstrfmon (char *s, size_t maxsize, const char *format, va_list ap)
{
  return __vstrfmon_l_internal (s, maxsize, _NL_CURRENT_LOCALE, format, ap,
				STRFMON_LDBL_IS_DBL);
}

ssize_t
attribute_compat_text_section
__nldbl___vstrfmon_l (char *s, size_t maxsize, locale_t loc,
		      const char *format, va_list ap)
{
  return __vstrfmon_l_internal (s, maxsize, loc, format, ap,
				STRFMON_LDBL_IS_DBL);
}

void
attribute_compat_text_section
__nldbl_syslog (int pri, const char *fmt, ...)
{
  va_list ap;
  va_start (ap, fmt);
  __vsyslog_internal (pri, fmt, ap, PRINTF_LDBL_IS_DBL);
  va_end (ap);
}

void
attribute_compat_text_section
__nldbl_vsyslog (int pri, const char *fmt, va_list ap)
{
  __vsyslog_internal (pri, fmt, ap, PRINTF_LDBL_IS_DBL);
}

void
attribute_compat_text_section
__nldbl___syslog_chk (int pri, int flag, const char *fmt, ...)
{
  va_list ap;
  unsigned int mode = PRINTF_LDBL_IS_DBL;
  if (flag > 0)
    mode |= PRINTF_FORTIFY;

  va_start (ap, fmt);
  __vsyslog_internal (pri, fmt, ap, mode);
  va_end(ap);
}

void
attribute_compat_text_section
__nldbl___vsyslog_chk (int pri, int flag, const char *fmt, va_list ap)
{
  unsigned int mode = PRINTF_LDBL_IS_DBL;
  if (flag > 0)
    mode |= PRINTF_FORTIFY;

  __vsyslog_internal (pri, fmt, ap, mode);
}

int
attribute_compat_text_section
__nldbl___isoc99_vfscanf (FILE *s, const char *fmt, va_list ap)
{
  return __vfscanf_internal (s, fmt, ap, SCANF_LDBL_IS_DBL | SCANF_ISOC99_A);
}
libc_hidden_def (__nldbl___isoc99_vfscanf)

int
attribute_compat_text_section
__nldbl___isoc99_sscanf (const char *s, const char *fmt, ...)
{
  _IO_strfile sf;
  FILE *f = _IO_strfile_read (&sf, s);
  va_list ap;
  int ret;

  va_start (ap, fmt);
  ret = __vfscanf_internal (f, fmt, ap, SCANF_LDBL_IS_DBL | SCANF_ISOC99_A);
  va_end (ap);

  return ret;
}

int
attribute_compat_text_section
__nldbl___isoc99_vsscanf (const char *s, const char *fmt, va_list ap)
{
  _IO_strfile sf;
  FILE *f = _IO_strfile_read (&sf, s);

  return __vfscanf_internal (f, fmt, ap, SCANF_LDBL_IS_DBL | SCANF_ISOC99_A);
}
libc_hidden_def (__nldbl___isoc99_vsscanf)

int
attribute_compat_text_section
__nldbl___isoc99_vscanf (const char *fmt, va_list ap)
{
  return __vfscanf_internal (stdin, fmt, ap,
			     SCANF_LDBL_IS_DBL | SCANF_ISOC99_A);
}

int
attribute_compat_text_section
__nldbl___isoc99_fscanf (FILE *s, const char *fmt, ...)
{
  va_list ap;
  int ret;

  va_start (ap, fmt);
  ret = __vfscanf_internal (s, fmt, ap, SCANF_LDBL_IS_DBL | SCANF_ISOC99_A);
  va_end (ap);

  return ret;
}

int
attribute_compat_text_section
__nldbl___isoc99_scanf (const char *fmt, ...)
{
  va_list ap;
  int ret;

  va_start (ap, fmt);
  ret = __vfscanf_internal (stdin, fmt, ap,
			    SCANF_LDBL_IS_DBL | SCANF_ISOC99_A);
  va_end (ap);

  return ret;
}

int
attribute_compat_text_section
__nldbl___isoc99_vfwscanf (FILE *s, const wchar_t *fmt, va_list ap)
{
  return __vfwscanf_internal (s, fmt, ap, SCANF_LDBL_IS_DBL | SCANF_ISOC99_A);
}
libc_hidden_def (__nldbl___isoc99_vfwscanf)

int
attribute_compat_text_section
__nldbl___isoc99_swscanf (const wchar_t *s, const wchar_t *fmt, ...)
{
  _IO_strfile sf;
  struct _IO_wide_data wd;
  FILE *f = _IO_strfile_readw (&sf, &wd, s);
  va_list ap;
  int ret;

  va_start (ap, fmt);
  ret = __vfwscanf_internal (f, fmt, ap, SCANF_LDBL_IS_DBL | SCANF_ISOC99_A);
  va_end (ap);

  return ret;
}

int
attribute_compat_text_section
__nldbl___isoc99_vswscanf (const wchar_t *s, const wchar_t *fmt, va_list ap)
{
  _IO_strfile sf;
  struct _IO_wide_data wd;
  FILE *f = _IO_strfile_readw (&sf, &wd, s);

  return __vfwscanf_internal (f, fmt, ap, SCANF_LDBL_IS_DBL | SCANF_ISOC99_A);
}
libc_hidden_def (__nldbl___isoc99_vswscanf)

int
attribute_compat_text_section
__nldbl___isoc99_vwscanf (const wchar_t *fmt, va_list ap)
{
  return __vfwscanf_internal (stdin, fmt, ap,
			     SCANF_LDBL_IS_DBL | SCANF_ISOC99_A);
}

int
attribute_compat_text_section
__nldbl___isoc99_fwscanf (FILE *s, const wchar_t *fmt, ...)
{
  va_list ap;
  int ret;

  va_start (ap, fmt);
  ret = __vfwscanf_internal (s, fmt, ap, SCANF_LDBL_IS_DBL | SCANF_ISOC99_A);
  va_end (ap);

  return ret;
}

int
attribute_compat_text_section
__nldbl___isoc99_wscanf (const wchar_t *fmt, ...)
{
  va_list ap;
  int ret;

  va_start (ap, fmt);
  ret = __vfwscanf_internal (stdin, fmt, ap,
			     SCANF_LDBL_IS_DBL | SCANF_ISOC99_A);
  va_end (ap);

  return ret;
}

void
__nldbl_argp_error (const struct argp_state *state, const char *fmt, ...)
{
  va_list ap;
  va_start (ap, fmt);
  __argp_error_internal (state, fmt, ap, PRINTF_LDBL_IS_DBL);
  va_end (ap);
}

void
__nldbl_argp_failure (const struct argp_state *state, int status,
			int errnum, const char *fmt, ...)
{
  va_list ap;
  va_start (ap, fmt);
  __argp_failure_internal (state, status, errnum, fmt, ap,
			   PRINTF_LDBL_IS_DBL);
  va_end (ap);
}

#define VA_CALL(call)							\
{									\
  va_list ap;								\
  va_start (ap, format);						\
  call (format, ap, PRINTF_LDBL_IS_DBL);				\
  va_end (ap);								\
}

void
__nldbl_err (int status, const char *format, ...)
{
  VA_CALL (__vwarn_internal)
  exit (status);
}

void
__nldbl_errx (int status, const char *format, ...)
{
  VA_CALL (__vwarnx_internal)
  exit (status);
}

void
__nldbl_verr (int status, const char *format, __gnuc_va_list ap)
{
  __vwarn_internal (format, ap, PRINTF_LDBL_IS_DBL);
  exit (status);
}

void
__nldbl_verrx (int status, const char *format, __gnuc_va_list ap)
{
  __vwarnx_internal (format, ap, PRINTF_LDBL_IS_DBL);
  exit (status);
}

void
__nldbl_warn (const char *format, ...)
{
  VA_CALL (__vwarn_internal)
}

void
__nldbl_warnx (const char *format, ...)
{
  VA_CALL (__vwarnx_internal)
}

void
__nldbl_vwarn (const char *format, __gnuc_va_list ap)
{
  __vwarn_internal (format, ap, PRINTF_LDBL_IS_DBL);
}

void
__nldbl_vwarnx (const char *format, __gnuc_va_list ap)
{
  __vwarnx_internal (format, ap, PRINTF_LDBL_IS_DBL);
}

void
__nldbl_error (int status, int errnum, const char *message, ...)
{
  va_list ap;
  va_start (ap, message);
  __error_internal (status, errnum, message, ap, PRINTF_LDBL_IS_DBL);
  va_end (ap);
}

void
__nldbl_error_at_line (int status, int errnum, const char *file_name,
		       unsigned int line_number, const char *message,
		       ...)
{
  va_list ap;
  va_start (ap, message);
  __error_at_line_internal (status, errnum, file_name, line_number,
			    message, ap, PRINTF_LDBL_IS_DBL);
  va_end (ap);
}

#if LONG_DOUBLE_COMPAT(libc, GLIBC_2_0)
compat_symbol (libc, __nldbl__IO_printf, _IO_printf, GLIBC_2_0);
compat_symbol (libc, __nldbl__IO_sprintf, _IO_sprintf, GLIBC_2_0);
compat_symbol (libc, __nldbl__IO_vfprintf, _IO_vfprintf, GLIBC_2_0);
compat_symbol (libc, __nldbl__IO_vsprintf, _IO_vsprintf, GLIBC_2_0);
compat_symbol (libc, __nldbl_dprintf, dprintf, GLIBC_2_0);
compat_symbol (libc, __nldbl_fprintf, fprintf, GLIBC_2_0);
compat_symbol (libc, __nldbl_printf, printf, GLIBC_2_0);
compat_symbol (libc, __nldbl_sprintf, sprintf, GLIBC_2_0);
compat_symbol (libc, __nldbl_vfprintf, vfprintf, GLIBC_2_0);
compat_symbol (libc, __nldbl_vprintf, vprintf, GLIBC_2_0);
compat_symbol (libc, __nldbl__IO_fprintf, _IO_fprintf, GLIBC_2_0);
compat_symbol (libc, __nldbl___vsnprintf, __vsnprintf, GLIBC_2_0);
compat_symbol (libc, __nldbl_asprintf, asprintf, GLIBC_2_0);
compat_symbol (libc, __nldbl_obstack_printf, obstack_printf, GLIBC_2_0);
compat_symbol (libc, __nldbl_obstack_vprintf, obstack_vprintf, GLIBC_2_0);
compat_symbol (libc, __nldbl_snprintf, snprintf, GLIBC_2_0);
compat_symbol (libc, __nldbl_vasprintf, vasprintf, GLIBC_2_0);
compat_symbol (libc, __nldbl_vdprintf, vdprintf, GLIBC_2_0);
compat_symbol (libc, __nldbl_vsnprintf, vsnprintf, GLIBC_2_0);
compat_symbol (libc, __nldbl_vsprintf, vsprintf, GLIBC_2_0);
compat_symbol (libc, __nldbl__IO_sscanf, _IO_sscanf, GLIBC_2_0);
compat_symbol (libc, __nldbl___vfscanf, __vfscanf, GLIBC_2_0);
compat_symbol (libc, __nldbl___vsscanf, __vsscanf, GLIBC_2_0);
compat_symbol (libc, __nldbl_fscanf, fscanf, GLIBC_2_0);
compat_symbol (libc, __nldbl_scanf, scanf, GLIBC_2_0);
compat_symbol (libc, __nldbl_sscanf, sscanf, GLIBC_2_0);
compat_symbol (libc, __nldbl_vfscanf, vfscanf, GLIBC_2_0);
compat_symbol (libc, __nldbl_vscanf, vscanf, GLIBC_2_0);
compat_symbol (libc, __nldbl_vsscanf, vsscanf, GLIBC_2_0);
compat_symbol (libc, __nldbl___printf_fp, __printf_fp, GLIBC_2_0);
compat_symbol (libc, __nldbl_strfmon, strfmon, GLIBC_2_0);
compat_symbol (libc, __nldbl_syslog, syslog, GLIBC_2_0);
compat_symbol (libc, __nldbl_vsyslog, vsyslog, GLIBC_2_0);
/* This function is not in public headers, but was exported until
   version 2.29.  For platforms that are newer than that, there's no
   need to expose the symbol.  */
# if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_29)
compat_symbol (libc, __nldbl__IO_vfscanf, _IO_vfscanf, GLIBC_2_0);
# endif
#endif
#if LONG_DOUBLE_COMPAT(libc, GLIBC_2_1)
compat_symbol (libc, __nldbl___asprintf, __asprintf, GLIBC_2_1);
compat_symbol (libc, __nldbl_printf_size, printf_size, GLIBC_2_1);
compat_symbol (libc, __nldbl___strfmon_l, __strfmon_l, GLIBC_2_1);
#endif
#if LONG_DOUBLE_COMPAT(libc, GLIBC_2_2)
compat_symbol (libc, __nldbl_swprintf, swprintf, GLIBC_2_2);
compat_symbol (libc, __nldbl_vwprintf, vwprintf, GLIBC_2_2);
compat_symbol (libc, __nldbl_wprintf, wprintf, GLIBC_2_2);
compat_symbol (libc, __nldbl_fwprintf, fwprintf, GLIBC_2_2);
compat_symbol (libc, __nldbl_vfwprintf, vfwprintf, GLIBC_2_2);
compat_symbol (libc, __nldbl_vswprintf, vswprintf, GLIBC_2_2);
compat_symbol (libc, __nldbl_fwscanf, fwscanf, GLIBC_2_2);
compat_symbol (libc, __nldbl_swscanf, swscanf, GLIBC_2_2);
compat_symbol (libc, __nldbl_vfwscanf, vfwscanf, GLIBC_2_2);
compat_symbol (libc, __nldbl_vswscanf, vswscanf, GLIBC_2_2);
compat_symbol (libc, __nldbl_vwscanf, vwscanf, GLIBC_2_2);
compat_symbol (libc, __nldbl_wscanf, wscanf, GLIBC_2_2);
#endif
#if LONG_DOUBLE_COMPAT(libc, GLIBC_2_3)
compat_symbol (libc, __nldbl_strfmon_l, strfmon_l, GLIBC_2_3);
#endif
#if LONG_DOUBLE_COMPAT(libc, GLIBC_2_3_4)
compat_symbol (libc, __nldbl___sprintf_chk, __sprintf_chk, GLIBC_2_3_4);
compat_symbol (libc, __nldbl___vsprintf_chk, __vsprintf_chk, GLIBC_2_3_4);
compat_symbol (libc, __nldbl___snprintf_chk, __snprintf_chk, GLIBC_2_3_4);
compat_symbol (libc, __nldbl___vsnprintf_chk, __vsnprintf_chk, GLIBC_2_3_4);
compat_symbol (libc, __nldbl___printf_chk, __printf_chk, GLIBC_2_3_4);
compat_symbol (libc, __nldbl___fprintf_chk, __fprintf_chk, GLIBC_2_3_4);
compat_symbol (libc, __nldbl___vprintf_chk, __vprintf_chk, GLIBC_2_3_4);
compat_symbol (libc, __nldbl___vfprintf_chk, __vfprintf_chk, GLIBC_2_3_4);
#endif
