/* Convert a floating-point number to string.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

/* Generic implementation for strfrom functions.  The implementation is generic
   for several floating-point types (e.g.: float, double), so that each
   function, such as strfromf and strfroml, share the same code, thus avoiding
   code duplication.  */

#include <ctype.h>
#include "../libio/libioP.h"
#include "../libio/strfile.h"
#include <printf.h>
#include <string.h>
#include <locale/localeinfo.h>

#define UCHAR_T char
#define L_(Str) Str
#define ISDIGIT(Ch) isdigit (Ch)
#include "stdio-common/printf-parse.h"

int
STRFROM (char *dest, size_t size, const char *format, FLOAT f)
{
  _IO_strnfile sfile;
#ifdef _IO_MTSAFE_IO
  sfile.f._sbf._f._lock = NULL;
#endif

  int done;

  /* Single-precision values need to be stored in a double type, because
     __printf_fp_l and __printf_fphex do not accept the float type.  */
  union {
    double flt;
    FLOAT value;
  } fpnum;
  const void *fpptr;
  fpptr = &fpnum;

  /* Variables to control the output format.  */
  int precision = -1; /* printf_fp and printf_fphex treat this internally.  */
  int specifier;
  struct printf_info info;

  /* Single-precision values need to be converted into double-precision,
     because __printf_fp and __printf_fphex only accept double and long double
     as the floating-point argument.  */
  if (__builtin_types_compatible_p (FLOAT, float))
    fpnum.flt = f;
  else
    fpnum.value = f;

  /* Check if the first character in the format string is indeed the '%'
     character.  Otherwise, abort.  */
  if (*format == '%')
    format++;
  else
    abort ();

  /* The optional precision specification always starts with a '.'.  If such
     character is present, read the precision.  */
  if (*format == '.')
    {
      format++;

      /* Parse the precision.  */
      if (ISDIGIT (*format))
	precision = read_int (&format);
      /* If only the period is specified, the precision is taken as zero, as
	 described in ISO/IEC 9899:2011, section 7.21.6.1, 4th paragraph, 3rd
	 item.  */
      else
	precision = 0;
    }

  /* Now there is only the conversion specifier to be read.  */
  switch (*format)
    {
    case 'a':
    case 'A':
    case 'e':
    case 'E':
    case 'f':
    case 'F':
    case 'g':
    case 'G':
      specifier = *format;
      break;
    default:
      abort ();
    }

  /* The following code to prepare the virtual file has been adapted from the
     function __vsnprintf_internal from libio.  */

  if (size == 0)
    {
    /* When size is zero, nothing is written and dest may be a null pointer.
       This is specified for snprintf in ISO/IEC 9899:2011, Section 7.21.6.5,
       in the second paragraph.  Thus, if size is zero, prepare to use the
       overflow buffer right from the start.  */
      dest = sfile.overflow_buf;
      size = sizeof (sfile.overflow_buf);
    }

  /* Prepare the virtual string file.  */
  _IO_no_init (&sfile.f._sbf._f, _IO_USER_LOCK, -1, NULL, NULL);
  _IO_JUMPS (&sfile.f._sbf) = &_IO_strn_jumps;
  _IO_str_init_static_internal (&sfile.f, dest, size - 1, dest);

  /* Prepare the format specification for printf_fp.  */
  memset (&info, '\0', sizeof (info));

  /* The functions strfromd and strfromf pass a floating-point number with
     double precision to printf_fp, whereas strfroml passes a floating-point
     number with long double precision.  The following line informs printf_fp
     which type of floating-point number is being passed.  */
  info.is_long_double = __builtin_types_compatible_p (FLOAT, long double);

  /* Similarly, the function strfromf128 passes a floating-point number in
     _Float128 format to printf_fp.  */
#if __HAVE_DISTINCT_FLOAT128
  info.is_binary128 = __builtin_types_compatible_p (FLOAT, _Float128);
#endif

  /* Set info according to the format string.  */
  info.prec = precision;
  info.spec = specifier;

  if (info.spec != 'a' && info.spec != 'A')
    done = __printf_fp_l (&sfile.f._sbf._f, _NL_CURRENT_LOCALE, &info, &fpptr);
  else
    done = __printf_fphex (&sfile.f._sbf._f, &info, &fpptr);

  /* Terminate the string.  */
  if (sfile.f._sbf._f._IO_buf_base != sfile.overflow_buf)
    *sfile.f._sbf._f._IO_write_ptr = '\0';

  return done;
}
