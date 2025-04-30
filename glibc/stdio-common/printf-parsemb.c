/* Helper functions for parsing printf format strings.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
   This file is part of th GNU C Library.

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

#include <ctype.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>
#include <wchar.h>
#include <wctype.h>

#ifndef COMPILE_WPRINTF
# define CHAR_T		char
# define UCHAR_T	unsigned char
# define INT_T		int
# define L_(Str)	Str
# define ISDIGIT(Ch)	isdigit (Ch)
# define HANDLE_REGISTERED_MODIFIER __handle_registered_modifier_mb
#else
# define CHAR_T		wchar_t
# define UCHAR_T	unsigned int
# define INT_T		wint_t
# define L_(Str)	L##Str
# define ISDIGIT(Ch)	iswdigit (Ch)
# define HANDLE_REGISTERED_MODIFIER __handle_registered_modifier_wc
#endif

#include "printf-parse.h"

#define NDEBUG 1
#include <assert.h>



/* FORMAT must point to a '%' at the beginning of a spec.  Fills in *SPEC
   with the parsed details.  POSN is the number of arguments already
   consumed.  At most MAXTYPES - POSN types are filled in TYPES.  Return
   the number of args consumed by this spec; *MAX_REF_ARG is updated so it
   remains the highest argument index used.  */
size_t
attribute_hidden
#ifdef COMPILE_WPRINTF
__parse_one_specwc (const UCHAR_T *format, size_t posn,
		    struct printf_spec *spec, size_t *max_ref_arg)
#else
__parse_one_specmb (const UCHAR_T *format, size_t posn,
		    struct printf_spec *spec, size_t *max_ref_arg)
#endif
{
  unsigned int n;
  size_t nargs = 0;

  /* Skip the '%'.  */
  ++format;

  /* Clear information structure.  */
  spec->data_arg = -1;
  spec->info.alt = 0;
  spec->info.space = 0;
  spec->info.left = 0;
  spec->info.showsign = 0;
  spec->info.group = 0;
  spec->info.i18n = 0;
  spec->info.extra = 0;
  spec->info.pad = ' ';
  spec->info.wide = sizeof (UCHAR_T) > 1;
  spec->info.is_binary128 = 0;

  /* Test for positional argument.  */
  if (ISDIGIT (*format))
    {
      const UCHAR_T *begin = format;

      n = read_int (&format);

      if (n != 0 && *format == L_('$'))
	/* Is positional parameter.  */
	{
	  ++format;		/* Skip the '$'.  */
	  if (n != -1)
	    {
	      spec->data_arg = n - 1;
	      *max_ref_arg = MAX (*max_ref_arg, n);
	    }
	}
      else
	/* Oops; that was actually the width and/or 0 padding flag.
	   Step back and read it again.  */
	format = begin;
    }

  /* Check for spec modifiers.  */
  do
    {
      switch (*format)
	{
	case L_(' '):
	  /* Output a space in place of a sign, when there is no sign.  */
	  spec->info.space = 1;
	  continue;
	case L_('+'):
	  /* Always output + or - for numbers.  */
	  spec->info.showsign = 1;
	  continue;
	case L_('-'):
	  /* Left-justify things.  */
	  spec->info.left = 1;
	  continue;
	case L_('#'):
	  /* Use the "alternate form":
	     Hex has 0x or 0X, FP always has a decimal point.  */
	  spec->info.alt = 1;
	  continue;
	case L_('0'):
	  /* Pad with 0s.  */
	  spec->info.pad = '0';
	  continue;
	case L_('\''):
	  /* Show grouping in numbers if the locale information
	     indicates any.  */
	  spec->info.group = 1;
	  continue;
	case L_('I'):
	  /* Use the internationalized form of the output.  Currently
	     means to use the `outdigits' of the current locale.  */
	  spec->info.i18n = 1;
	  continue;
	default:
	  break;
	}
      break;
    }
  while (*++format);

  if (spec->info.left)
    spec->info.pad = ' ';

  /* Get the field width.  */
  spec->width_arg = -1;
  spec->info.width = 0;
  if (*format == L_('*'))
    {
      /* The field width is given in an argument.
	 A negative field width indicates left justification.  */
      const UCHAR_T *begin = ++format;

      if (ISDIGIT (*format))
	{
	  /* The width argument might be found in a positional parameter.  */
	  n = read_int (&format);

	  if (n != 0 && *format == L_('$'))
	    {
	      if (n != -1)
		{
		  spec->width_arg = n - 1;
		  *max_ref_arg = MAX (*max_ref_arg, n);
		}
	      ++format;		/* Skip '$'.  */
	    }
	}

      if (spec->width_arg < 0)
	{
	  /* Not in a positional parameter.  Consume one argument.  */
	  spec->width_arg = posn++;
	  ++nargs;
	  format = begin;	/* Step back and reread.  */
	}
    }
  else if (ISDIGIT (*format))
    {
      int n = read_int (&format);

      /* Constant width specification.  */
      if (n != -1)
	spec->info.width = n;
    }
  /* Get the precision.  */
  spec->prec_arg = -1;
  /* -1 means none given; 0 means explicit 0.  */
  spec->info.prec = -1;
  if (*format == L_('.'))
    {
      ++format;
      if (*format == L_('*'))
	{
	  /* The precision is given in an argument.  */
	  const UCHAR_T *begin = ++format;

	  if (ISDIGIT (*format))
	    {
	      n = read_int (&format);

	      if (n != 0 && *format == L_('$'))
		{
		  if (n != -1)
		    {
		      spec->prec_arg = n - 1;
		      *max_ref_arg = MAX (*max_ref_arg, n);
		    }
		  ++format;
		}
	    }

	  if (spec->prec_arg < 0)
	    {
	      /* Not in a positional parameter.  */
	      spec->prec_arg = posn++;
	      ++nargs;
	      format = begin;
	    }
	}
      else if (ISDIGIT (*format))
	{
	  int n = read_int (&format);

	  if (n != -1)
	    spec->info.prec = n;
	}
      else
	/* "%.?" is treated like "%.0?".  */
	spec->info.prec = 0;
    }

  /* Check for type modifiers.  */
  spec->info.is_long_double = 0;
  spec->info.is_short = 0;
  spec->info.is_long = 0;
  spec->info.is_char = 0;
  spec->info.user = 0;

  if (__builtin_expect (__printf_modifier_table == NULL, 1)
      || __printf_modifier_table[*format] == NULL
      || HANDLE_REGISTERED_MODIFIER (&format, &spec->info) != 0)
    switch (*format++)
      {
      case L_('h'):
	/* ints are short ints or chars.  */
	if (*format != L_('h'))
	  spec->info.is_short = 1;
	else
	  {
	    ++format;
	    spec->info.is_char = 1;
	  }
	break;
      case L_('l'):
	/* ints are long ints.  */
	spec->info.is_long = 1;
	if (*format != L_('l'))
	  break;
	++format;
	/* FALLTHROUGH */
      case L_('L'):
	/* doubles are long doubles, and ints are long long ints.  */
      case L_('q'):
	/* 4.4 uses this for long long.  */
	spec->info.is_long_double = 1;
	break;
      case L_('z'):
      case L_('Z'):
	/* ints are size_ts.  */
	assert (sizeof (size_t) <= sizeof (unsigned long long int));
#if LONG_MAX != LONG_LONG_MAX
	spec->info.is_long_double = (sizeof (size_t)
				     > sizeof (unsigned long int));
#endif
	spec->info.is_long = sizeof (size_t) > sizeof (unsigned int);
	break;
      case L_('t'):
	assert (sizeof (ptrdiff_t) <= sizeof (long long int));
#if LONG_MAX != LONG_LONG_MAX
	spec->info.is_long_double = (sizeof (ptrdiff_t) > sizeof (long int));
#endif
	spec->info.is_long = sizeof (ptrdiff_t) > sizeof (int);
	break;
      case L_('j'):
	assert (sizeof (uintmax_t) <= sizeof (unsigned long long int));
#if LONG_MAX != LONG_LONG_MAX
	spec->info.is_long_double = (sizeof (uintmax_t)
				     > sizeof (unsigned long int));
#endif
	spec->info.is_long = sizeof (uintmax_t) > sizeof (unsigned int);
	break;
      default:
	/* Not a recognized modifier.  Backup.  */
	--format;
	break;
      }

  /* Get the format specification.  */
  spec->info.spec = (wchar_t) *format++;
  spec->size = -1;
  if (__builtin_expect (__printf_function_table == NULL, 1)
      || spec->info.spec > UCHAR_MAX
      || __printf_arginfo_table[spec->info.spec] == NULL
      /* We don't try to get the types for all arguments if the format
	 uses more than one.  The normal case is covered though.  If
	 the call returns -1 we continue with the normal specifiers.  */
      || (int) (spec->ndata_args = (*__printf_arginfo_table[spec->info.spec])
				   (&spec->info, 1, &spec->data_arg_type,
				    &spec->size)) < 0)
    {
      /* Find the data argument types of a built-in spec.  */
      spec->ndata_args = 1;

      switch (spec->info.spec)
	{
	case L'i':
	case L'd':
	case L'u':
	case L'o':
	case L'X':
	case L'x':
#if LONG_MAX != LONG_LONG_MAX
	  if (spec->info.is_long_double)
	    spec->data_arg_type = PA_INT|PA_FLAG_LONG_LONG;
	  else
#endif
	    if (spec->info.is_long)
	      spec->data_arg_type = PA_INT|PA_FLAG_LONG;
	    else if (spec->info.is_short)
	      spec->data_arg_type = PA_INT|PA_FLAG_SHORT;
	    else if (spec->info.is_char)
	      spec->data_arg_type = PA_CHAR;
	    else
	      spec->data_arg_type = PA_INT;
	  break;
	case L'e':
	case L'E':
	case L'f':
	case L'F':
	case L'g':
	case L'G':
	case L'a':
	case L'A':
	  if (spec->info.is_long_double)
	    spec->data_arg_type = PA_DOUBLE|PA_FLAG_LONG_DOUBLE;
	  else
	    spec->data_arg_type = PA_DOUBLE;
	  break;
	case L'c':
	  spec->data_arg_type = PA_CHAR;
	  break;
	case L'C':
	  spec->data_arg_type = PA_WCHAR;
	  break;
	case L's':
	  spec->data_arg_type = PA_STRING;
	  break;
	case L'S':
	  spec->data_arg_type = PA_WSTRING;
	  break;
	case L'p':
	  spec->data_arg_type = PA_POINTER;
	  break;
	case L'n':
	  spec->data_arg_type = PA_INT|PA_FLAG_PTR;
	  break;

	case L'm':
	default:
	  /* An unknown spec will consume no args.  */
	  spec->ndata_args = 0;
	  break;
	}
    }

  if (spec->data_arg == -1 && spec->ndata_args > 0)
    {
      /* There are args consumed, but no positional spec.  Use the
	 next sequential arg position.  */
      spec->data_arg = posn;
      nargs += spec->ndata_args;
    }

  if (spec->info.spec == L'\0')
    /* Format ended before this spec was complete.  */
    spec->end_of_fmt = spec->next_fmt = format - 1;
  else
    {
      /* Find the next format spec.  */
      spec->end_of_fmt = format;
#ifdef COMPILE_WPRINTF
      spec->next_fmt = __find_specwc (format);
#else
      spec->next_fmt = __find_specmb (format);
#endif
    }

  return nargs;
}
