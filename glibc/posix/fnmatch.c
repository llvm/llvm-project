/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#ifndef _LIBC
# include <libc-config.h>
#endif

/* Enable GNU extensions in fnmatch.h.  */
#ifndef _GNU_SOURCE
# define _GNU_SOURCE    1
#endif

#include <fnmatch.h>

#include <assert.h>
#include <errno.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#if defined _LIBC || HAVE_ALLOCA
# include <alloca.h>
#endif
#include <wchar.h>
#include <wctype.h>
#include <stddef.h>
#include <stdbool.h>

/* We need some of the locale data (the collation sequence information)
   but there is no interface to get this information in general.  Therefore
   we support a correct implementation only in glibc.  */
#ifdef _LIBC
# include "../locale/localeinfo.h"
# include "../locale/coll-lookup.h"
# include <shlib-compat.h>

# define CONCAT(a,b) __CONCAT(a,b)
# define btowc __btowc
# define iswctype __iswctype
# define mbsrtowcs __mbsrtowcs
# define mempcpy __mempcpy
# define strnlen __strnlen
# define towlower __towlower
# define wcscat __wcscat
# define wcslen __wcslen
# define wctype __wctype
# define wmemchr __wmemchr
# define wmempcpy __wmempcpy
# define fnmatch __fnmatch
extern int fnmatch (const char *pattern, const char *string, int flags);
#endif

#ifdef _LIBC
# if __GNUC__ >= 7
#  define FALLTHROUGH __attribute__ ((__fallthrough__))
# else
#  define FALLTHROUGH ((void) 0)
# endif
#else
# include "attribute.h"
#endif

#include <intprops.h>
#include <flexmember.h>
#include <scratch_buffer.h>

#ifdef _LIBC
typedef ptrdiff_t idx_t;
#else
# include "idx.h"
#endif

/* We often have to test for FNM_FILE_NAME and FNM_PERIOD being both set.  */
#define NO_LEADING_PERIOD(flags) \
  ((flags & (FNM_FILE_NAME | FNM_PERIOD)) == (FNM_FILE_NAME | FNM_PERIOD))

#ifndef _LIBC
# if HAVE_ALLOCA
/* The OS usually guarantees only one guard page at the bottom of the stack,
   and a page size can be as small as 4096 bytes.  So we cannot safely
   allocate anything larger than 4096 bytes.  Also care for the possibility
   of a few compiler-allocated temporary stack slots.  */
#  define __libc_use_alloca(n) ((n) < 4032)
# else
/* Just use malloc.  */
#  define __libc_use_alloca(n) false
#  undef alloca
#  define alloca(n) malloc (n)
# endif
# define alloca_account(size, avar) ((avar) += (size), alloca (size))
#endif

/* Provide support for user-defined character classes, based on the functions
   from ISO C 90 amendment 1.  */
#ifdef CHARCLASS_NAME_MAX
# define CHAR_CLASS_MAX_LENGTH CHARCLASS_NAME_MAX
#else
/* This shouldn't happen but some implementation might still have this
   problem.  Use a reasonable default value.  */
# define CHAR_CLASS_MAX_LENGTH 256
#endif

#define IS_CHAR_CLASS(string) wctype (string)

/* Avoid depending on library functions or files
   whose names are inconsistent.  */

/* Global variable.  */
static int posixly_correct;

/* Note that this evaluates C many times.  */
#define FOLD(c) ((flags & FNM_CASEFOLD) ? tolower (c) : (c))
#define CHAR    char
#define UCHAR   unsigned char
#define INT     int
#define FCT     internal_fnmatch
#define EXT     ext_match
#define END     end_pattern
#define STRUCT  fnmatch_struct
#define L_(CS)  CS
#define BTOWC(C) btowc (C)
#define STRLEN(S) strlen (S)
#define STRCAT(D, S) strcat (D, S)
#define MEMPCPY(D, S, N) mempcpy (D, S, N)
#define MEMCHR(S, C, N) memchr (S, C, N)
#define WIDE_CHAR_VERSION 0
#ifdef _LIBC
# include <locale/weight.h>
# define FINDIDX findidx
#endif
#include "fnmatch_loop.c"


#define FOLD(c) ((flags & FNM_CASEFOLD) ? towlower (c) : (c))
#define CHAR    wchar_t
#define UCHAR   wint_t
#define INT     wint_t
#define FCT     internal_fnwmatch
#define EXT     ext_wmatch
#define END     end_wpattern
#define L_(CS)  L##CS
#define BTOWC(C) (C)
#define STRLEN(S) wcslen (S)
#define STRCAT(D, S) wcscat (D, S)
#define MEMPCPY(D, S, N) wmempcpy (D, S, N)
#define MEMCHR(S, C, N) wmemchr (S, C, N)
#define WIDE_CHAR_VERSION 1
#ifdef _LIBC
/* Change the name the header defines so it doesn't conflict with
   the <locale/weight.h> version included above.  */
# define findidx findidxwc
# include <locale/weightwc.h>
# undef findidx
# define FINDIDX findidxwc
#endif

#undef IS_CHAR_CLASS
/* We have to convert the wide character string in a multibyte string.  But
   we know that the character class names consist of alphanumeric characters
   from the portable character set, and since the wide character encoding
   for a member of the portable character set is the same code point as
   its single-byte encoding, we can use a simplified method to convert the
   string to a multibyte character string.  */
static wctype_t
is_char_class (const wchar_t *wcs)
{
  char s[CHAR_CLASS_MAX_LENGTH + 1];
  char *cp = s;

  do
    {
      /* Test for a printable character from the portable character set.  */
#ifdef _LIBC
      if (*wcs < 0x20 || *wcs > 0x7e
          || *wcs == 0x24 || *wcs == 0x40 || *wcs == 0x60)
        return (wctype_t) 0;
#else
      switch (*wcs)
        {
        case L' ': case L'!': case L'"': case L'#': case L'%':
        case L'&': case L'\'': case L'(': case L')': case L'*':
        case L'+': case L',': case L'-': case L'.': case L'/':
        case L'0': case L'1': case L'2': case L'3': case L'4':
        case L'5': case L'6': case L'7': case L'8': case L'9':
        case L':': case L';': case L'<': case L'=': case L'>':
        case L'?':
        case L'A': case L'B': case L'C': case L'D': case L'E':
        case L'F': case L'G': case L'H': case L'I': case L'J':
        case L'K': case L'L': case L'M': case L'N': case L'O':
        case L'P': case L'Q': case L'R': case L'S': case L'T':
        case L'U': case L'V': case L'W': case L'X': case L'Y':
        case L'Z':
        case L'[': case L'\\': case L']': case L'^': case L'_':
        case L'a': case L'b': case L'c': case L'd': case L'e':
        case L'f': case L'g': case L'h': case L'i': case L'j':
        case L'k': case L'l': case L'm': case L'n': case L'o':
        case L'p': case L'q': case L'r': case L's': case L't':
        case L'u': case L'v': case L'w': case L'x': case L'y':
        case L'z': case L'{': case L'|': case L'}': case L'~':
          break;
        default:
          return (wctype_t) 0;
        }
#endif

      /* Avoid overrunning the buffer.  */
      if (cp == s + CHAR_CLASS_MAX_LENGTH)
        return (wctype_t) 0;

      *cp++ = (char) *wcs++;
    }
  while (*wcs != L'\0');

  *cp = '\0';

  return wctype (s);
}
#define IS_CHAR_CLASS(string) is_char_class (string)

#include "fnmatch_loop.c"

static int
fnmatch_convert_to_wide (const char *str, struct scratch_buffer *buf,
                         size_t *n)
{
  mbstate_t ps;
  memset (&ps, '\0', sizeof (ps));

  size_t nw = buf->length / sizeof (wchar_t);
  *n = strnlen (str, nw - 1);
  if (__glibc_likely (*n < nw))
    {
      const char *p = str;
      *n = mbsrtowcs (buf->data, &p, *n + 1, &ps);
      if (__glibc_unlikely (*n == (size_t) -1))
        /* Something wrong.
           XXX Do we have to set 'errno' to something which mbsrtows hasn't
           already done?  */
        return -1;
      if (p == NULL)
        return 0;
      memset (&ps, '\0', sizeof (ps));
    }

  *n = mbsrtowcs (NULL, &str, 0, &ps);
  if (__glibc_unlikely (*n == (size_t) -1))
    return -1;
  if (!scratch_buffer_set_array_size (buf, *n + 1, sizeof (wchar_t)))
    {
      __set_errno (ENOMEM);
      return -2;
    }
  assert (mbsinit (&ps));
  mbsrtowcs (buf->data, &str, *n + 1, &ps);
  return 0;
}

int
fnmatch (const char *pattern, const char *string, int flags)
{
  if (__glibc_unlikely (MB_CUR_MAX != 1))
    {
      size_t n;
      struct scratch_buffer wpattern;
      scratch_buffer_init (&wpattern);
      struct scratch_buffer wstring;
      scratch_buffer_init (&wstring);
      int r;

      /* Convert the strings into wide characters.  Any conversion issue
         fallback to the ascii version.  */
      r = fnmatch_convert_to_wide (pattern, &wpattern, &n);
      if (r == 0)
        {
          r = fnmatch_convert_to_wide (string, &wstring, &n);
          if (r == 0)
            r = internal_fnwmatch (wpattern.data, wstring.data,
                                   (wchar_t *) wstring.data + n,
                                   flags & FNM_PERIOD, flags, NULL,
                                   false);
        }

      scratch_buffer_free (&wstring);
      scratch_buffer_free (&wpattern);

      if (r == -2 || r == 0)
        return r;
    }

  return internal_fnmatch (pattern, string, string + strlen (string),
                           flags & FNM_PERIOD, flags, NULL, 0);
}

#undef fnmatch
versioned_symbol (libc, __fnmatch, fnmatch, GLIBC_2_2_3);
#if SHLIB_COMPAT(libc, GLIBC_2_0, GLIBC_2_2_3)
strong_alias (__fnmatch, __fnmatch_old)
compat_symbol (libc, __fnmatch_old, fnmatch, GLIBC_2_0);
#endif
libc_hidden_ver (__fnmatch, fnmatch)
