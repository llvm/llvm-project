/* Internal functions for the *scanf* implementation.
   Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <ctype.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <wctype.h>
#include <libc-diag.h>
#include <libc-lock.h>
#include <locale/localeinfo.h>
#include <scratch_buffer.h>

#ifdef	__GNUC__
# define HAVE_LONGLONG
# define LONGLONG	long long
#else
# define LONGLONG	long
#endif

/* Determine whether we have to handle `long long' at all.  */
#if LONG_MAX == LONG_LONG_MAX
# define need_longlong	0
#else
# define need_longlong	1
#endif

/* Determine whether we have to handle `long'.  */
#if INT_MAX == LONG_MAX
# define need_long	0
#else
# define need_long	1
#endif

/* Those are flags in the conversion format. */
#define LONG		0x0001	/* l: long or double */
#define LONGDBL		0x0002	/* L: long long or long double */
#define SHORT		0x0004	/* h: short */
#define SUPPRESS	0x0008	/* *: suppress assignment */
#define POINTER		0x0010	/* weird %p pointer (`fake hex') */
#define NOSKIP		0x0020	/* do not skip blanks */
#define NUMBER_SIGNED	0x0040	/* signed integer */
#define GROUP		0x0080	/* ': group numbers */
#define GNU_MALLOC	0x0100	/* a: malloc strings */
#define CHAR		0x0200	/* hh: char */
#define I18N		0x0400	/* I: use locale's digits */
#define HEXA_FLOAT	0x0800	/* hexadecimal float */
#define READ_POINTER	0x1000	/* this is a pointer value */
#define POSIX_MALLOC	0x2000	/* m: malloc strings */
#define MALLOC		(GNU_MALLOC | POSIX_MALLOC)

#include <locale/localeinfo.h>
#include <libioP.h>

#ifdef COMPILE_WSCANF
# define ungetc(c, s)	((void) (c == WEOF				      \
				 || (--read_in,				      \
				     _IO_sputbackwc (s, c))))
# define ungetc_not_eof(c, s)	((void) (--read_in,			      \
					 _IO_sputbackwc (s, c)))
# define inchar()	(c == WEOF ? ((errno = inchar_errno), WEOF)	      \
			 : ((c = _IO_getwc_unlocked (s)),		      \
			    (void) (c != WEOF				      \
				    ? ++read_in				      \
				    : (size_t) (inchar_errno = errno)), c))

# define ISSPACE(Ch)	  iswspace (Ch)
# define ISDIGIT(Ch)	  iswdigit (Ch)
# define ISXDIGIT(Ch)	  iswxdigit (Ch)
# define TOLOWER(Ch)	  towlower (Ch)
# define ORIENT	  if (_IO_fwide (s, 1) != 1) return WEOF
# define __strtoll_internal	__wcstoll_internal
# define __strtoull_internal	__wcstoull_internal
# define __strtol_internal	__wcstol_internal
# define __strtoul_internal	__wcstoul_internal
# define __strtold_internal	__wcstold_internal
# define __strtod_internal	__wcstod_internal
# define __strtof_internal	__wcstof_internal
# if __HAVE_FLOAT128_UNLIKE_LDBL
#  define __strtof128_internal	__wcstof128_internal
# endif

# define L_(Str)	L##Str
# define CHAR_T		wchar_t
# define UCHAR_T	unsigned int
# define WINT_T		wint_t
# undef EOF
# define EOF		WEOF
#else
# define ungetc(c, s)	((void) ((int) c == EOF				      \
				 || (--read_in,				      \
				     _IO_sputbackc (s, (unsigned char) c))))
# define ungetc_not_eof(c, s)	((void) (--read_in,			      \
					 _IO_sputbackc (s, (unsigned char) c)))
# define inchar()	(c == EOF ? ((errno = inchar_errno), EOF)	      \
			 : ((c = _IO_getc_unlocked (s)),		      \
			    (void) (c != EOF				      \
				    ? ++read_in				      \
				    : (size_t) (inchar_errno = errno)), c))
# define ISSPACE(Ch)	  __isspace_l (Ch, loc)
# define ISDIGIT(Ch)	  __isdigit_l (Ch, loc)
# define ISXDIGIT(Ch)	  __isxdigit_l (Ch, loc)
# define TOLOWER(Ch)	  __tolower_l ((unsigned char) (Ch), loc)
# define ORIENT	  if (_IO_vtable_offset (s) == 0			      \
			      && _IO_fwide (s, -1) != -1)		      \
			    return EOF

# define L_(Str)	Str
# define CHAR_T		char
# define UCHAR_T	unsigned char
# define WINT_T		int
#endif

#include "printf-parse.h" /* Use read_int.  */

#define encode_error() do {						      \
			  __set_errno (EILSEQ);				      \
			  goto errout;					      \
			} while (0)
#define conv_error()	do {						      \
			  goto errout;					      \
			} while (0)
#define input_error()	do {						      \
			  if (done == 0) done = EOF;			      \
			  goto errout;					      \
			} while (0)
#define add_ptr_to_free(ptr)						      \
  do									      \
    {									      \
      if (ptrs_to_free == NULL						      \
	  || ptrs_to_free->count == (sizeof (ptrs_to_free->ptrs)	      \
				     / sizeof (ptrs_to_free->ptrs[0])))	      \
	{								      \
	  struct ptrs_to_free *new_ptrs = alloca (sizeof (*ptrs_to_free));    \
	  new_ptrs->count = 0;						      \
	  new_ptrs->next = ptrs_to_free;				      \
	  ptrs_to_free = new_ptrs;					      \
	}								      \
      ptrs_to_free->ptrs[ptrs_to_free->count++] = (ptr);		      \
    }									      \
  while (0)
#define ARGCHECK(s, format)						      \
  do									      \
    {									      \
      /* Check file argument for consistence.  */			      \
      CHECK_FILE (s, EOF);						      \
      if (s->_flags & _IO_NO_READS)					      \
	{								      \
	  __set_errno (EBADF);						      \
	  return EOF;							      \
	}								      \
      else if (format == NULL)						      \
	{								      \
	  __set_errno (EINVAL);						      \
	  return EOF;							      \
	}								      \
    } while (0)
#define LOCK_STREAM(S)							      \
  __libc_cleanup_region_start (1, (void (*) (void *)) &_IO_funlockfile, (S)); \
  _IO_flockfile (S)
#define UNLOCK_STREAM(S)						      \
  _IO_funlockfile (S);							      \
  __libc_cleanup_region_end (0)

struct ptrs_to_free
{
  size_t count;
  struct ptrs_to_free *next;
  char **ptrs[32];
};

struct char_buffer {
  CHAR_T *current;
  CHAR_T *end;
  struct scratch_buffer scratch;
};

/* Returns a pointer to the first CHAR_T object in the buffer.  Only
   valid if char_buffer_add (BUFFER, CH) has been called and
   char_buffer_error (BUFFER) is false.  */
static inline CHAR_T *
char_buffer_start (const struct char_buffer *buffer)
{
  return (CHAR_T *) buffer->scratch.data;
}

/* Returns the number of CHAR_T objects in the buffer.  Only valid if
   char_buffer_error (BUFFER) is false.  */
static inline size_t
char_buffer_size (const struct char_buffer *buffer)
{
  return buffer->current - char_buffer_start (buffer);
}

/* Reinitializes BUFFER->current and BUFFER->end to cover the entire
   scratch buffer.  */
static inline void
char_buffer_rewind (struct char_buffer *buffer)
{
  buffer->current = char_buffer_start (buffer);
  buffer->end = buffer->current + buffer->scratch.length / sizeof (CHAR_T);
}

/* Returns true if a previous call to char_buffer_add (BUFFER, CH)
   failed.  */
static inline bool
char_buffer_error (const struct char_buffer *buffer)
{
  return __glibc_unlikely (buffer->current == NULL);
}

/* Slow path for char_buffer_add.  */
static void
char_buffer_add_slow (struct char_buffer *buffer, CHAR_T ch)
{
  if (char_buffer_error (buffer))
    return;
  size_t offset = buffer->end - (CHAR_T *) buffer->scratch.data;
  if (!scratch_buffer_grow_preserve (&buffer->scratch))
    {
      buffer->current = NULL;
      buffer->end = NULL;
      return;
    }
  char_buffer_rewind (buffer);
  buffer->current += offset;
  *buffer->current++ = ch;
}

/* Adds CH to BUFFER.  This function does not report any errors, check
   for them with char_buffer_error.  */
static inline void
char_buffer_add (struct char_buffer *buffer, CHAR_T ch)
  __attribute__ ((always_inline));
static inline void
char_buffer_add (struct char_buffer *buffer, CHAR_T ch)
{
  if (__glibc_unlikely (buffer->current == buffer->end))
    char_buffer_add_slow (buffer, ch);
  else
    *buffer->current++ = ch;
}

/* Read formatted input from S according to the format string
   FORMAT, using the argument list in ARG.
   Return the number of assignments made, or -1 for an input error.  */
#ifdef COMPILE_WSCANF
int
__vfwscanf_internal (FILE *s, const wchar_t *format, va_list argptr,
		     unsigned int mode_flags)
#else
int
__vfscanf_internal (FILE *s, const char *format, va_list argptr,
		    unsigned int mode_flags)
#endif
{
  va_list arg;
  const UCHAR_T *f = (const UCHAR_T *) format;
  UCHAR_T fc;	/* Current character of the format.  */
  WINT_T done = 0;	/* Assignments done.  */
  size_t read_in = 0;	/* Chars read in.  */
  WINT_T c = 0;	/* Last char read.  */
  int width;		/* Maximum field width.  */
  int flags;		/* Modifiers for current format element.  */
#ifndef COMPILE_WSCANF
  locale_t loc = _NL_CURRENT_LOCALE;
  struct __locale_data *const curctype = loc->__locales[LC_CTYPE];
#endif

  /* Errno of last failed inchar call.  */
  int inchar_errno = 0;
  /* Status for reading F-P nums.  */
  char got_digit, got_dot, got_e, got_sign;
  /* If a [...] is a [^...].  */
  CHAR_T not_in;
#define exp_char not_in
  /* Base for integral numbers.  */
  int base;
  /* Decimal point character.  */
#ifdef COMPILE_WSCANF
  wint_t decimal;
#else
  const char *decimal;
#endif
  /* The thousands character of the current locale.  */
#ifdef COMPILE_WSCANF
  wint_t thousands;
#else
  const char *thousands;
#endif
  struct ptrs_to_free *ptrs_to_free = NULL;
  /* State for the conversions.  */
  mbstate_t state;
  /* Integral holding variables.  */
  union
    {
      long long int q;
      unsigned long long int uq;
      long int l;
      unsigned long int ul;
    } num;
  /* Character-buffer pointer.  */
  char *str = NULL;
  wchar_t *wstr = NULL;
  char **strptr = NULL;
  ssize_t strsize = 0;
  /* We must not react on white spaces immediately because they can
     possibly be matched even if in the input stream no character is
     available anymore.  */
  int skip_space = 0;
  /* Workspace.  */
  CHAR_T *tw;			/* Temporary pointer.  */
  struct char_buffer charbuf;
  scratch_buffer_init (&charbuf.scratch);

#ifdef __va_copy
  __va_copy (arg, argptr);
#else
  arg = (va_list) argptr;
#endif

#ifdef ORIENT
  ORIENT;
#endif

  ARGCHECK (s, format);

 {
#ifndef COMPILE_WSCANF
   struct __locale_data *const curnumeric = loc->__locales[LC_NUMERIC];
#endif

   /* Figure out the decimal point character.  */
#ifdef COMPILE_WSCANF
   decimal = _NL_CURRENT_WORD (LC_NUMERIC, _NL_NUMERIC_DECIMAL_POINT_WC);
#else
   decimal = curnumeric->values[_NL_ITEM_INDEX (DECIMAL_POINT)].string;
#endif
   /* Figure out the thousands separator character.  */
#ifdef COMPILE_WSCANF
   thousands = _NL_CURRENT_WORD (LC_NUMERIC, _NL_NUMERIC_THOUSANDS_SEP_WC);
#else
   thousands = curnumeric->values[_NL_ITEM_INDEX (THOUSANDS_SEP)].string;
   if (*thousands == '\0')
     thousands = NULL;
#endif
 }

  /* Lock the stream.  */
  LOCK_STREAM (s);


#ifndef COMPILE_WSCANF
  /* From now on we use `state' to convert the format string.  */
  memset (&state, '\0', sizeof (state));
#endif

  /* Run through the format string.  */
  while (*f != '\0')
    {
      unsigned int argpos;
      /* Extract the next argument, which is of type TYPE.
	 For a %N$... spec, this is the Nth argument from the beginning;
	 otherwise it is the next argument after the state now in ARG.  */
#ifdef __va_copy
# define ARG(type)	(argpos == 0 ? va_arg (arg, type)		      \
			 : ({ unsigned int pos = argpos;		      \
			      va_list arg;				      \
			      __va_copy (arg, argptr);			      \
			      while (--pos > 0)				      \
				(void) va_arg (arg, void *);		      \
			      va_arg (arg, type);			      \
			    }))
#else
# if 0
      /* XXX Possible optimization.  */
#  define ARG(type)	(argpos == 0 ? va_arg (arg, type)		      \
			 : ({ va_list arg = (va_list) argptr;		      \
			      arg = (va_list) ((char *) arg		      \
					       + (argpos - 1)		      \
					       * __va_rounded_size (void *)); \
			      va_arg (arg, type);			      \
			   }))
# else
#  define ARG(type)	(argpos == 0 ? va_arg (arg, type)		      \
			 : ({ unsigned int pos = argpos;		      \
			      va_list arg = (va_list) argptr;		      \
			      while (--pos > 0)				      \
				(void) va_arg (arg, void *);		      \
			      va_arg (arg, type);			      \
			    }))
# endif
#endif

#ifndef COMPILE_WSCANF
      if (!isascii (*f))
	{
	  /* Non-ASCII, may be a multibyte.  */
	  int len = __mbrlen ((const char *) f, strlen ((const char *) f),
			      &state);
	  if (len > 0)
	    {
	      do
		{
		  c = inchar ();
		  if (__glibc_unlikely (c == EOF))
		    input_error ();
		  else if (c != *f++)
		    {
		      ungetc_not_eof (c, s);
		      conv_error ();
		    }
		}
	      while (--len > 0);
	      continue;
	    }
	}
#endif

      fc = *f++;
      if (fc != '%')
	{
	  /* Remember to skip spaces.  */
	  if (ISSPACE (fc))
	    {
	      skip_space = 1;
	      continue;
	    }

	  /* Read a character.  */
	  c = inchar ();

	  /* Characters other than format specs must just match.  */
	  if (__glibc_unlikely (c == EOF))
	    input_error ();

	  /* We saw white space char as the last character in the format
	     string.  Now it's time to skip all leading white space.  */
	  if (skip_space)
	    {
	      while (ISSPACE (c))
		if (__glibc_unlikely (inchar () == EOF))
		  input_error ();
	      skip_space = 0;
	    }

	  if (__glibc_unlikely (c != fc))
	    {
	      ungetc (c, s);
	      conv_error ();
	    }

	  continue;
	}

      /* This is the start of the conversion string. */
      flags = 0;

      /* Initialize state of modifiers.  */
      argpos = 0;

      /* Prepare temporary buffer.  */
      char_buffer_rewind (&charbuf);

      /* Check for a positional parameter specification.  */
      if (ISDIGIT (*f))
	{
	  argpos = read_int (&f);
	  if (*f == L_('$'))
	    ++f;
	  else
	    {
	      /* Oops; that was actually the field width.  */
	      width = argpos;
	      argpos = 0;
	      goto got_width;
	    }
	}

      /* Check for the assignment-suppressing, the number grouping flag,
	 and the signal to use the locale's digit representation.  */
      while (*f == L_('*') || *f == L_('\'') || *f == L_('I'))
	switch (*f++)
	  {
	  case L_('*'):
	    flags |= SUPPRESS;
	    break;
	  case L_('\''):
#ifdef COMPILE_WSCANF
	    if (thousands != L'\0')
#else
	    if (thousands != NULL)
#endif
	      flags |= GROUP;
	    break;
	  case L_('I'):
	    flags |= I18N;
	    break;
	  }

      /* Find the maximum field width.  */
      width = 0;
      if (ISDIGIT (*f))
	width = read_int (&f);
    got_width:
      if (width == 0)
	width = -1;

      /* Check for type modifiers.  */
      switch (*f++)
	{
	case L_('h'):
	  /* ints are short ints or chars.  */
	  if (*f == L_('h'))
	    {
	      ++f;
	      flags |= CHAR;
	    }
	  else
	    flags |= SHORT;
	  break;
	case L_('l'):
	  if (*f == L_('l'))
	    {
	      /* A double `l' is equivalent to an `L'.  */
	      ++f;
	      flags |= LONGDBL | LONG;
	    }
	  else
	    /* ints are long ints.  */
	    flags |= LONG;
	  break;
	case L_('q'):
	case L_('L'):
	  /* doubles are long doubles, and ints are long long ints.  */
	  flags |= LONGDBL | LONG;
	  break;
	case L_('a'):
	  /* The `a' is used as a flag only if followed by `s', `S' or
	     `['.  */
	  if (*f != L_('s') && *f != L_('S') && *f != L_('['))
	    {
	      --f;
	      break;
	    }
	  /* In __isoc99_*scanf %as, %aS and %a[ extension is not
	     supported at all.  */
	  if (__glibc_likely ((mode_flags & SCANF_ISOC99_A) != 0))
	    {
	      --f;
	      break;
	    }
	  /* String conversions (%s, %[) take a `char **'
	     arg and fill it in with a malloc'd pointer.  */
	  flags |= GNU_MALLOC;
	  break;
	case L_('m'):
	  flags |= POSIX_MALLOC;
	  if (*f == L_('l'))
	    {
	      ++f;
	      flags |= LONG;
	    }
	  break;
	case L_('z'):
	  if (need_longlong && sizeof (size_t) > sizeof (unsigned long int))
	    flags |= LONGDBL;
	  else if (sizeof (size_t) > sizeof (unsigned int))
	    flags |= LONG;
	  break;
	case L_('j'):
	  if (need_longlong && sizeof (uintmax_t) > sizeof (unsigned long int))
	    flags |= LONGDBL;
	  else if (sizeof (uintmax_t) > sizeof (unsigned int))
	    flags |= LONG;
	  break;
	case L_('t'):
	  if (need_longlong && sizeof (ptrdiff_t) > sizeof (long int))
	    flags |= LONGDBL;
	  else if (sizeof (ptrdiff_t) > sizeof (int))
	    flags |= LONG;
	  break;
	default:
	  /* Not a recognized modifier.  Backup.  */
	  --f;
	  break;
	}

      /* End of the format string?  */
      if (__glibc_unlikely (*f == L_('\0')))
	conv_error ();

      /* Find the conversion specifier.  */
      fc = *f++;
      if (skip_space || (fc != L_('[') && fc != L_('c')
			 && fc != L_('C') && fc != L_('n')))
	{
	  /* Eat whitespace.  */
	  int save_errno = errno;
	  __set_errno (0);
	  do
	    /* We add the additional test for EOF here since otherwise
	       inchar will restore the old errno value which might be
	       EINTR but does not indicate an interrupt since nothing
	       was read at this time.  */
	    if (__builtin_expect ((c == EOF || inchar () == EOF)
				  && errno == EINTR, 0))
	      input_error ();
	  while (ISSPACE (c));
	  __set_errno (save_errno);
	  ungetc (c, s);
	  skip_space = 0;
	}

      switch (fc)
	{
	case L_('%'):	/* Must match a literal '%'.  */
	  c = inchar ();
	  if (__glibc_unlikely (c == EOF))
	    input_error ();
	  if (__glibc_unlikely (c != fc))
	    {
	      ungetc_not_eof (c, s);
	      conv_error ();
	    }
	  break;

	case L_('n'):	/* Answer number of assignments done.  */
	  /* Corrigendum 1 to ISO C 1990 describes the allowed flags
	     with the 'n' conversion specifier.  */
	  if (!(flags & SUPPRESS))
	    {
	      /* Don't count the read-ahead.  */
	      if (need_longlong && (flags & LONGDBL))
		*ARG (long long int *) = read_in;
	      else if (need_long && (flags & LONG))
		*ARG (long int *) = read_in;
	      else if (flags & SHORT)
		*ARG (short int *) = read_in;
	      else if (!(flags & CHAR))
		*ARG (int *) = read_in;
	      else
		*ARG (char *) = read_in;

#ifdef NO_BUG_IN_ISO_C_CORRIGENDUM_1
	      /* We have a severe problem here.  The ISO C standard
		 contradicts itself in explaining the effect of the %n
		 format in `scanf'.  While in ISO C:1990 and the ISO C
		 Amendement 1:1995 the result is described as

		   Execution of a %n directive does not effect the
		   assignment count returned at the completion of
		   execution of the f(w)scanf function.

		 in ISO C Corrigendum 1:1994 the following was added:

		   Subclause 7.9.6.2
		   Add the following fourth example:
		     In:
		       #include <stdio.h>
		       int d1, d2, n1, n2, i;
		       i = sscanf("123", "%d%n%n%d", &d1, &n1, &n2, &d2);
		     the value 123 is assigned to d1 and the value3 to n1.
		     Because %n can never get an input failure the value
		     of 3 is also assigned to n2.  The value of d2 is not
		     affected.  The value 3 is assigned to i.

		 We go for now with the historically correct code from ISO C,
		 i.e., we don't count the %n assignments.  When it ever
		 should proof to be wrong just remove the #ifdef above.  */
	      ++done;
#endif
	    }
	  break;

	case L_('c'):	/* Match characters.  */
	  if ((flags & LONG) == 0)
	    {
	      if (width == -1)
		width = 1;

#define STRING_ARG(Str, Type, Width)					      \
	      do if (!(flags & SUPPRESS))				      \
		{							      \
		  if (flags & MALLOC)					      \
		    {							      \
		      /* The string is to be stored in a malloc'd buffer.  */ \
		      /* For %mS using char ** is actually wrong, but	      \
			 shouldn't make a difference on any arch glibc	      \
			 supports and would unnecessarily complicate	      \
			 things. */					      \
		      strptr = ARG (char **);				      \
		      if (strptr == NULL)				      \
			conv_error ();					      \
		      /* Allocate an initial buffer.  */		      \
		      strsize = Width;					      \
		      *strptr = (char *) malloc (strsize * sizeof (Type));    \
		      Str = (Type *) *strptr;				      \
		      if (Str != NULL)					      \
			add_ptr_to_free (strptr);			      \
		      else if (flags & POSIX_MALLOC)			      \
			{						      \
			  done = EOF;					      \
			  goto errout;					      \
			}						      \
		    }							      \
		  else							      \
		    Str = ARG (Type *);					      \
		  if (Str == NULL)					      \
		    conv_error ();					      \
		} while (0)
#ifdef COMPILE_WSCANF
	      STRING_ARG (str, char, 100);
#else
	      STRING_ARG (str, char, (width > 1024 ? 1024 : width));
#endif

	      c = inchar ();
	      if (__glibc_unlikely (c == EOF))
		input_error ();

#ifdef COMPILE_WSCANF
	      /* We have to convert the wide character(s) into multibyte
		 characters and store the result.  */
	      memset (&state, '\0', sizeof (state));

	      do
		{
		  size_t n;

		  if (!(flags & SUPPRESS) && (flags & POSIX_MALLOC)
		      && *strptr + strsize - str <= MB_LEN_MAX)
		    {
		      /* We have to enlarge the buffer if the `m' flag
			 was given.  */
		      size_t strleng = str - *strptr;
		      char *newstr;

		      newstr = (char *) realloc (*strptr, strsize * 2);
		      if (newstr == NULL)
			{
			  /* Can't allocate that much.  Last-ditch effort.  */
			  newstr = (char *) realloc (*strptr,
						     strleng + MB_LEN_MAX);
			  if (newstr == NULL)
			    {
			      /* c can't have `a' flag, only `m'.  */
			      done = EOF;
			      goto errout;
			    }
			  else
			    {
			      *strptr = newstr;
			      str = newstr + strleng;
			      strsize = strleng + MB_LEN_MAX;
			    }
			}
		      else
			{
			  *strptr = newstr;
			  str = newstr + strleng;
			  strsize *= 2;
			}
		    }

		  n = __wcrtomb (!(flags & SUPPRESS) ? str : NULL, c, &state);
		  if (__glibc_unlikely (n == (size_t) -1))
		    /* No valid wide character.  */
		    input_error ();

		  /* Increment the output pointer.  Even if we don't
		     write anything.  */
		  str += n;
		}
	      while (--width > 0 && inchar () != EOF);
#else
	      if (!(flags & SUPPRESS))
		{
		  do
		    {
		      if ((flags & MALLOC)
			  && (char *) str == *strptr + strsize)
			{
			  /* Enlarge the buffer.  */
			  size_t newsize
			    = strsize
			      + (strsize >= width ? width - 1 : strsize);

			  str = (char *) realloc (*strptr, newsize);
			  if (str == NULL)
			    {
			      /* Can't allocate that much.  Last-ditch
				 effort.  */
			      str = (char *) realloc (*strptr, strsize + 1);
			      if (str == NULL)
				{
				  /* c can't have `a' flag, only `m'.  */
				  done = EOF;
				  goto errout;
				}
			      else
				{
				  *strptr = (char *) str;
				  str += strsize;
				  ++strsize;
				}
			    }
			  else
			    {
			      *strptr = (char *) str;
			      str += strsize;
			      strsize = newsize;
			    }
			}
		      *str++ = c;
		    }
		  while (--width > 0 && inchar () != EOF);
		}
	      else
		while (--width > 0 && inchar () != EOF);
#endif

	      if (!(flags & SUPPRESS))
		{
		  if ((flags & MALLOC) && str - *strptr != strsize)
		    {
		      char *cp = (char *) realloc (*strptr, str - *strptr);
		      if (cp != NULL)
			*strptr = cp;
		    }
		  strptr = NULL;
		  ++done;
		}

	      break;
	    }
	  /* FALLTHROUGH */
	case L_('C'):
	  if (width == -1)
	    width = 1;

	  STRING_ARG (wstr, wchar_t, (width > 1024 ? 1024 : width));

	  c = inchar ();
	  if (__glibc_unlikely (c == EOF))
	    input_error ();

#ifdef COMPILE_WSCANF
	  /* Just store the incoming wide characters.  */
	  if (!(flags & SUPPRESS))
	    {
	      do
		{
		  if ((flags & MALLOC)
		      && wstr == (wchar_t *) *strptr + strsize)
		    {
		      size_t newsize
			= strsize + (strsize > width ? width - 1 : strsize);
		      /* Enlarge the buffer.  */
		      wstr = (wchar_t *) realloc (*strptr,
						  newsize * sizeof (wchar_t));
		      if (wstr == NULL)
			{
			  /* Can't allocate that much.  Last-ditch effort.  */
			  wstr = (wchar_t *) realloc (*strptr,
						      (strsize + 1)
						      * sizeof (wchar_t));
			  if (wstr == NULL)
			    {
			      /* C or lc can't have `a' flag, only `m'
				 flag.  */
			      done = EOF;
			      goto errout;
			    }
			  else
			    {
			      *strptr = (char *) wstr;
			      wstr += strsize;
			      ++strsize;
			    }
			}
		      else
			{
			  *strptr = (char *) wstr;
			  wstr += strsize;
			  strsize = newsize;
			}
		    }
		  *wstr++ = c;
		}
	      while (--width > 0 && inchar () != EOF);
	    }
	  else
	    while (--width > 0 && inchar () != EOF);
#else
	  {
	    /* We have to convert the multibyte input sequence to wide
	       characters.  */
	    char buf[1];
	    mbstate_t cstate;

	    memset (&cstate, '\0', sizeof (cstate));

	    do
	      {
		/* This is what we present the mbrtowc function first.  */
		buf[0] = c;

		if (!(flags & SUPPRESS) && (flags & MALLOC)
		    && wstr == (wchar_t *) *strptr + strsize)
		  {
		    size_t newsize
		      = strsize + (strsize > width ? width - 1 : strsize);
		    /* Enlarge the buffer.  */
		    wstr = (wchar_t *) realloc (*strptr,
						newsize * sizeof (wchar_t));
		    if (wstr == NULL)
		      {
			/* Can't allocate that much.  Last-ditch effort.  */
			wstr = (wchar_t *) realloc (*strptr,
						    ((strsize + 1)
						     * sizeof (wchar_t)));
			if (wstr == NULL)
			  {
			    /* C or lc can't have `a' flag, only `m' flag.  */
			    done = EOF;
			    goto errout;
			  }
			else
			  {
			    *strptr = (char *) wstr;
			    wstr += strsize;
			    ++strsize;
			  }
		      }
		    else
		      {
			*strptr = (char *) wstr;
			wstr += strsize;
			strsize = newsize;
		      }
		  }

		while (1)
		  {
		    size_t n;

		    n = __mbrtowc (!(flags & SUPPRESS) ? wstr : NULL,
				   buf, 1, &cstate);

		    if (n == (size_t) -2)
		      {
			/* Possibly correct character, just not enough
			   input.  */
			if (__glibc_unlikely (inchar () == EOF))
			  encode_error ();

			buf[0] = c;
			continue;
		      }

		    if (__glibc_unlikely (n != 1))
		      encode_error ();

		    /* We have a match.  */
		    break;
		  }

		/* Advance the result pointer.  */
		++wstr;
	      }
	    while (--width > 0 && inchar () != EOF);
	  }
#endif

	  if (!(flags & SUPPRESS))
	    {
	      if ((flags & MALLOC) && wstr - (wchar_t *) *strptr != strsize)
		{
		  wchar_t *cp = (wchar_t *) realloc (*strptr,
						     ((wstr
						       - (wchar_t *) *strptr)
						      * sizeof (wchar_t)));
		  if (cp != NULL)
		    *strptr = (char *) cp;
		}
	      strptr = NULL;

	      ++done;
	    }

	  break;

	case L_('s'):		/* Read a string.  */
	  if (!(flags & LONG))
	    {
	      STRING_ARG (str, char, 100);

	      c = inchar ();
	      if (__glibc_unlikely (c == EOF))
		input_error ();

#ifdef COMPILE_WSCANF
	      memset (&state, '\0', sizeof (state));
#endif

	      do
		{
		  if (ISSPACE (c))
		    {
		      ungetc_not_eof (c, s);
		      break;
		    }

#ifdef COMPILE_WSCANF
		  /* This is quite complicated.  We have to convert the
		     wide characters into multibyte characters and then
		     store them.  */
		  {
		    size_t n;

		    if (!(flags & SUPPRESS) && (flags & MALLOC)
			&& *strptr + strsize - str <= MB_LEN_MAX)
		      {
			/* We have to enlarge the buffer if the `a' or `m'
			   flag was given.  */
			size_t strleng = str - *strptr;
			char *newstr;

			newstr = (char *) realloc (*strptr, strsize * 2);
			if (newstr == NULL)
			  {
			    /* Can't allocate that much.  Last-ditch
			       effort.  */
			    newstr = (char *) realloc (*strptr,
						       strleng + MB_LEN_MAX);
			    if (newstr == NULL)
			      {
				if (flags & POSIX_MALLOC)
				  {
				    done = EOF;
				    goto errout;
				  }
				/* We lose.  Oh well.  Terminate the
				   string and stop converting,
				   so at least we don't skip any input.  */
				((char *) (*strptr))[strleng] = '\0';
				strptr = NULL;
				++done;
				conv_error ();
			      }
			    else
			      {
				*strptr = newstr;
				str = newstr + strleng;
				strsize = strleng + MB_LEN_MAX;
			      }
			  }
			else
			  {
			    *strptr = newstr;
			    str = newstr + strleng;
			    strsize *= 2;
			  }
		      }

		    n = __wcrtomb (!(flags & SUPPRESS) ? str : NULL, c,
				   &state);
		    if (__glibc_unlikely (n == (size_t) -1))
		      encode_error ();

		    assert (n <= MB_LEN_MAX);
		    str += n;
		  }
#else
		  /* This is easy.  */
		  if (!(flags & SUPPRESS))
		    {
		      *str++ = c;
		      if ((flags & MALLOC)
			  && (char *) str == *strptr + strsize)
			{
			  /* Enlarge the buffer.  */
			  str = (char *) realloc (*strptr, 2 * strsize);
			  if (str == NULL)
			    {
			      /* Can't allocate that much.  Last-ditch
				 effort.  */
			      str = (char *) realloc (*strptr, strsize + 1);
			      if (str == NULL)
				{
				  if (flags & POSIX_MALLOC)
				    {
				      done = EOF;
				      goto errout;
				    }
				  /* We lose.  Oh well.  Terminate the
				     string and stop converting,
				     so at least we don't skip any input.  */
				  ((char *) (*strptr))[strsize - 1] = '\0';
				  strptr = NULL;
				  ++done;
				  conv_error ();
				}
			      else
				{
				  *strptr = (char *) str;
				  str += strsize;
				  ++strsize;
				}
			    }
			  else
			    {
			      *strptr = (char *) str;
			      str += strsize;
			      strsize *= 2;
			    }
			}
		    }
#endif
		}
	      while ((width <= 0 || --width > 0) && inchar () != EOF);

	      if (!(flags & SUPPRESS))
		{
#ifdef COMPILE_WSCANF
		  /* We have to emit the code to get into the initial
		     state.  */
		  char buf[MB_LEN_MAX];
		  size_t n = __wcrtomb (buf, L'\0', &state);
		  if (n > 0 && (flags & MALLOC)
		      && str + n >= *strptr + strsize)
		    {
		      /* Enlarge the buffer.  */
		      size_t strleng = str - *strptr;
		      char *newstr;

		      newstr = (char *) realloc (*strptr, strleng + n + 1);
		      if (newstr == NULL)
			{
			  if (flags & POSIX_MALLOC)
			    {
			      done = EOF;
			      goto errout;
			    }
			  /* We lose.  Oh well.  Terminate the string
			     and stop converting, so at least we don't
			     skip any input.  */
			  ((char *) (*strptr))[strleng] = '\0';
			  strptr = NULL;
			  ++done;
			  conv_error ();
			}
		      else
			{
			  *strptr = newstr;
			  str = newstr + strleng;
			  strsize = strleng + n + 1;
			}
		    }

		  str = __mempcpy (str, buf, n);
#endif
		  *str++ = '\0';

		  if ((flags & MALLOC) && str - *strptr != strsize)
		    {
		      char *cp = (char *) realloc (*strptr, str - *strptr);
		      if (cp != NULL)
			*strptr = cp;
		    }
		  strptr = NULL;

		  ++done;
		}
	      break;
	    }
	  /* FALLTHROUGH */

	case L_('S'):
	  {
#ifndef COMPILE_WSCANF
	    mbstate_t cstate;
#endif

	    /* Wide character string.  */
	    STRING_ARG (wstr, wchar_t, 100);

	    c = inchar ();
	    if (__builtin_expect (c == EOF,  0))
	      input_error ();

#ifndef COMPILE_WSCANF
	    memset (&cstate, '\0', sizeof (cstate));
#endif

	    do
	      {
		if (ISSPACE (c))
		  {
		    ungetc_not_eof (c, s);
		    break;
		  }

#ifdef COMPILE_WSCANF
		/* This is easy.  */
		if (!(flags & SUPPRESS))
		  {
		    *wstr++ = c;
		    if ((flags & MALLOC)
			&& wstr == (wchar_t *) *strptr + strsize)
		      {
			/* Enlarge the buffer.  */
			wstr = (wchar_t *) realloc (*strptr,
						    (2 * strsize)
						    * sizeof (wchar_t));
			if (wstr == NULL)
			  {
			    /* Can't allocate that much.  Last-ditch
			       effort.  */
			    wstr = (wchar_t *) realloc (*strptr,
							(strsize + 1)
							* sizeof (wchar_t));
			    if (wstr == NULL)
			      {
				if (flags & POSIX_MALLOC)
				  {
				    done = EOF;
				    goto errout;
				  }
				/* We lose.  Oh well.  Terminate the string
				   and stop converting, so at least we don't
				   skip any input.  */
				((wchar_t *) (*strptr))[strsize - 1] = L'\0';
				strptr = NULL;
				++done;
				conv_error ();
			      }
			    else
			      {
				*strptr = (char *) wstr;
				wstr += strsize;
				++strsize;
			      }
			  }
			else
			  {
			    *strptr = (char *) wstr;
			    wstr += strsize;
			    strsize *= 2;
			  }
		      }
		  }
#else
		{
		  char buf[1];

		  buf[0] = c;

		  while (1)
		    {
		      size_t n;

		      n = __mbrtowc (!(flags & SUPPRESS) ? wstr : NULL,
				     buf, 1, &cstate);

		      if (n == (size_t) -2)
			{
			  /* Possibly correct character, just not enough
			     input.  */
			  if (__glibc_unlikely (inchar () == EOF))
			    encode_error ();

			  buf[0] = c;
			  continue;
			}

		      if (__glibc_unlikely (n != 1))
			encode_error ();

		      /* We have a match.  */
		      ++wstr;
		      break;
		    }

		  if (!(flags & SUPPRESS) && (flags & MALLOC)
		      && wstr == (wchar_t *) *strptr + strsize)
		    {
		      /* Enlarge the buffer.  */
		      wstr = (wchar_t *) realloc (*strptr,
						  (2 * strsize
						   * sizeof (wchar_t)));
		      if (wstr == NULL)
			{
			  /* Can't allocate that much.  Last-ditch effort.  */
			  wstr = (wchar_t *) realloc (*strptr,
						      ((strsize + 1)
						       * sizeof (wchar_t)));
			  if (wstr == NULL)
			    {
			      if (flags & POSIX_MALLOC)
				{
				  done = EOF;
				  goto errout;
				}
			      /* We lose.  Oh well.  Terminate the
				 string and stop converting, so at
				 least we don't skip any input.  */
			      ((wchar_t *) (*strptr))[strsize - 1] = L'\0';
			      strptr = NULL;
			      ++done;
			      conv_error ();
			    }
			  else
			    {
			      *strptr = (char *) wstr;
			      wstr += strsize;
			      ++strsize;
			    }
			}
		      else
			{
			  *strptr = (char *) wstr;
			  wstr += strsize;
			  strsize *= 2;
			}
		    }
		}
#endif
	      }
	    while ((width <= 0 || --width > 0) && inchar () != EOF);

	    if (!(flags & SUPPRESS))
	      {
		*wstr++ = L'\0';

		if ((flags & MALLOC) && wstr - (wchar_t *) *strptr != strsize)
		  {
		    wchar_t *cp = (wchar_t *) realloc (*strptr,
						       ((wstr
							 - (wchar_t *) *strptr)
							* sizeof (wchar_t)));
		    if (cp != NULL)
		      *strptr = (char *) cp;
		  }
		strptr = NULL;

		++done;
	      }
	  }
	  break;

	case L_('x'):	/* Hexadecimal integer.  */
	case L_('X'):	/* Ditto.  */
	  base = 16;
	  goto number;

	case L_('o'):	/* Octal integer.  */
	  base = 8;
	  goto number;

	case L_('u'):	/* Unsigned decimal integer.  */
	  base = 10;
	  goto number;

	case L_('d'):	/* Signed decimal integer.  */
	  base = 10;
	  flags |= NUMBER_SIGNED;
	  goto number;

	case L_('i'):	/* Generic number.  */
	  base = 0;
	  flags |= NUMBER_SIGNED;

	number:
	  c = inchar ();
	  if (__glibc_unlikely (c == EOF))
	    input_error ();

	  /* Check for a sign.  */
	  if (c == L_('-') || c == L_('+'))
	    {
	      char_buffer_add (&charbuf, c);
	      if (width > 0)
		--width;
	      c = inchar ();
	    }

	  /* Look for a leading indication of base.  */
	  if (width != 0 && c == L_('0'))
	    {
	      if (width > 0)
		--width;

	      char_buffer_add (&charbuf, c);
	      c = inchar ();

	      if (width != 0 && TOLOWER (c) == L_('x'))
		{
		  if (base == 0)
		    base = 16;
		  if (base == 16)
		    {
		      if (width > 0)
			--width;
		      c = inchar ();
		    }
		}
	      else if (base == 0)
		base = 8;
	    }

	  if (base == 0)
	    base = 10;

	  if (base == 10 && __builtin_expect ((flags & I18N) != 0, 0))
	    {
	      int from_level;
	      int to_level;
	      int level;
#ifdef COMPILE_WSCANF
	      const wchar_t *wcdigits[10];
	      const wchar_t *wcdigits_extended[10];
#else
	      const char *mbdigits[10];
	      const char *mbdigits_extended[10];
#endif
	      /*  "to_inpunct" is a map from ASCII digits to their
		  equivalent in locale. This is defined for locales
		  which use an extra digits set.  */
	      wctrans_t map = __wctrans ("to_inpunct");
	      int n;

	      from_level = 0;
#ifdef COMPILE_WSCANF
	      to_level = _NL_CURRENT_WORD (LC_CTYPE,
					   _NL_CTYPE_INDIGITS_WC_LEN) - 1;
#else
	      to_level = (uint32_t) curctype->values[_NL_ITEM_INDEX (_NL_CTYPE_INDIGITS_MB_LEN)].word - 1;
#endif

	      /* Get the alternative digit forms if there are any.  */
	      if (__glibc_unlikely (map != NULL))
		{
		  /*  Adding new level for extra digits set in locale file.  */
		  ++to_level;

		  for (n = 0; n < 10; ++n)
		    {
#ifdef COMPILE_WSCANF
		      wcdigits[n] = (const wchar_t *)
			_NL_CURRENT (LC_CTYPE, _NL_CTYPE_INDIGITS0_WC + n);

		      wchar_t *wc_extended = (wchar_t *)
			alloca ((to_level + 2) * sizeof (wchar_t));
		      __wmemcpy (wc_extended, wcdigits[n], to_level);
		      wc_extended[to_level] = __towctrans (L'0' + n, map);
		      wc_extended[to_level + 1] = '\0';
		      wcdigits_extended[n] = wc_extended;
#else
		      mbdigits[n]
			= curctype->values[_NL_CTYPE_INDIGITS0_MB + n].string;

		      /*  Get the equivalent wide char in map.  */
		      wint_t extra_wcdigit = __towctrans (L'0' + n, map);

		      /*  Convert it to multibyte representation.  */
		      mbstate_t state;
		      memset (&state, '\0', sizeof (state));

		      char extra_mbdigit[MB_LEN_MAX];
		      size_t mblen
			= __wcrtomb (extra_mbdigit, extra_wcdigit, &state);

		      if (mblen == (size_t) -1)
			{
			  /*  Ignore this new level.  */
			  map = NULL;
			  break;
			}

		      /*  Calculate the length of mbdigits[n].  */
		      const char *last_char = mbdigits[n];
		      for (level = 0; level < to_level; ++level)
			last_char = strchr (last_char, '\0') + 1;

		      size_t mbdigits_len = last_char - mbdigits[n];

		      /*  Allocate memory for extended multibyte digit.  */
		      char *mb_extended;
		      mb_extended = (char *) alloca (mbdigits_len + mblen + 1);

		      /*  And get the mbdigits + extra_digit string.  */
		      *(char *) __mempcpy (__mempcpy (mb_extended, mbdigits[n],
						      mbdigits_len),
					   extra_mbdigit, mblen) = '\0';
		      mbdigits_extended[n] = mb_extended;
#endif
		    }
		}

	      /* Read the number into workspace.  */
	      while (c != EOF && width != 0)
		{
		  /* In this round we get the pointer to the digit strings
		     and also perform the first round of comparisons.  */
		  for (n = 0; n < 10; ++n)
		    {
		      /* Get the string for the digits with value N.  */
#ifdef COMPILE_WSCANF

		      /* wcdigits_extended[] is fully set in the loop
			 above, but the test for "map != NULL" is done
			 inside the loop here and outside the loop there.  */
		      DIAG_PUSH_NEEDS_COMMENT;
#if defined(__clang__)
		      DIAG_IGNORE_NEEDS_COMMENT (4.7, "-Wsometimes-uninitialized");
#else
		      DIAG_IGNORE_NEEDS_COMMENT (4.7, "-Wmaybe-uninitialized");
#endif

		      if (__glibc_unlikely (map != NULL))
			wcdigits[n] = wcdigits_extended[n];
		      else
			wcdigits[n] = (const wchar_t *)
			  _NL_CURRENT (LC_CTYPE, _NL_CTYPE_INDIGITS0_WC + n);
		      wcdigits[n] += from_level;

		      DIAG_POP_NEEDS_COMMENT;

		      if (c == (wint_t) *wcdigits[n])
			{
			  to_level = from_level;
			  break;
			}

		      /* Advance the pointer to the next string.  */
		      ++wcdigits[n];
#else
		      const char *cmpp;
		      int avail = width > 0 ? width : INT_MAX;

		      if (__glibc_unlikely (map != NULL))
			mbdigits[n] = mbdigits_extended[n];
		      else
			mbdigits[n]
			  = curctype->values[_NL_CTYPE_INDIGITS0_MB + n].string;

		      for (level = 0; level < from_level; level++)
			mbdigits[n] = strchr (mbdigits[n], '\0') + 1;

		      cmpp = mbdigits[n];
		      while ((unsigned char) *cmpp == c && avail >= 0)
			{
			  if (*++cmpp == '\0')
			    break;
			  else
			    {
			      if (avail == 0 || inchar () == EOF)
				break;
			      --avail;
			    }
			}

		      if (*cmpp == '\0')
			{
			  if (width > 0)
			    width = avail;
			  to_level = from_level;
			  break;
			}

		      /* We are pushing all read characters back.  */
		      if (cmpp > mbdigits[n])
			{
			  ungetc (c, s);
			  while (--cmpp > mbdigits[n])
			    ungetc_not_eof ((unsigned char) *cmpp, s);
			  c = (unsigned char) *cmpp;
			}

		      /* Advance the pointer to the next string.  */
		      mbdigits[n] = strchr (mbdigits[n], '\0') + 1;
#endif
		    }

		  if (n == 10)
		    {
		      /* Have not yet found the digit.  */
		      for (level = from_level + 1; level <= to_level; ++level)
			{
			  /* Search all ten digits of this level.  */
			  for (n = 0; n < 10; ++n)
			    {
#ifdef COMPILE_WSCANF
			      if (c == (wint_t) *wcdigits[n])
				break;

			      /* Advance the pointer to the next string.  */
			      ++wcdigits[n];
#else
			      const char *cmpp;
			      int avail = width > 0 ? width : INT_MAX;

			      cmpp = mbdigits[n];
			      while ((unsigned char) *cmpp == c && avail >= 0)
				{
				  if (*++cmpp == '\0')
				    break;
				  else
				    {
				      if (avail == 0 || inchar () == EOF)
					break;
				      --avail;
				    }
				}

			      if (*cmpp == '\0')
				{
				  if (width > 0)
				    width = avail;
				  break;
				}

			      /* We are pushing all read characters back.  */
			      if (cmpp > mbdigits[n])
				{
				  ungetc (c, s);
				  while (--cmpp > mbdigits[n])
				    ungetc_not_eof ((unsigned char) *cmpp, s);
				  c = (unsigned char) *cmpp;
				}

			      /* Advance the pointer to the next string.  */
			      mbdigits[n] = strchr (mbdigits[n], '\0') + 1;
#endif
			    }

			  if (n < 10)
			    {
			      /* Found it.  */
			      from_level = level;
			      to_level = level;
			      break;
			    }
			}
		    }

		  if (n < 10)
		    c = L_('0') + n;
		  else if (flags & GROUP)
		    {
		      /* Try matching against the thousands separator.  */
#ifdef COMPILE_WSCANF
		      if (c != thousands)
			  break;
#else
		      const char *cmpp = thousands;
		      int avail = width > 0 ? width : INT_MAX;

		      while ((unsigned char) *cmpp == c && avail >= 0)
			{
			  char_buffer_add (&charbuf, c);
			  if (*++cmpp == '\0')
			    break;
			  else
			    {
			      if (avail == 0 || inchar () == EOF)
				break;
			      --avail;
			    }
			}

		      if (char_buffer_error (&charbuf))
			{
			  __set_errno (ENOMEM);
			  done = EOF;
			  goto errout;
			}

		      if (*cmpp != '\0')
			{
			  /* We are pushing all read characters back.  */
			  if (cmpp > thousands)
			    {
			      charbuf.current -= cmpp - thousands;
			      ungetc (c, s);
			      while (--cmpp > thousands)
				ungetc_not_eof ((unsigned char) *cmpp, s);
			      c = (unsigned char) *cmpp;
			    }
			  break;
			}

		      if (width > 0)
			width = avail;

		      /* The last thousands character will be added back by
			 the char_buffer_add below.  */
		      --charbuf.current;
#endif
		    }
		  else
		    break;

		  char_buffer_add (&charbuf, c);
		  if (width > 0)
		    --width;

		  c = inchar ();
		}
	    }
	  else
	    /* Read the number into workspace.  */
	    while (c != EOF && width != 0)
	      {
		if (base == 16)
		  {
		    if (!ISXDIGIT (c))
		      break;
		  }
		else if (!ISDIGIT (c) || (int) (c - L_('0')) >= base)
		  {
		    if (base == 10 && (flags & GROUP))
		      {
			/* Try matching against the thousands separator.  */
#ifdef COMPILE_WSCANF
			if (c != thousands)
			  break;
#else
			const char *cmpp = thousands;
			int avail = width > 0 ? width : INT_MAX;

			while ((unsigned char) *cmpp == c && avail >= 0)
			  {
			    char_buffer_add (&charbuf, c);
			    if (*++cmpp == '\0')
			      break;
			    else
			      {
				if (avail == 0 || inchar () == EOF)
				  break;
				--avail;
			      }
			  }

			if (char_buffer_error (&charbuf))
			  {
			    __set_errno (ENOMEM);
			    done = EOF;
			    goto errout;
			  }

			if (*cmpp != '\0')
			  {
			    /* We are pushing all read characters back.  */
			    if (cmpp > thousands)
			      {
				charbuf.current -= cmpp - thousands;
				ungetc (c, s);
				while (--cmpp > thousands)
				  ungetc_not_eof ((unsigned char) *cmpp, s);
				c = (unsigned char) *cmpp;
			      }
			    break;
			  }

			if (width > 0)
			  width = avail;

			/* The last thousands character will be added back by
			   the char_buffer_add below.  */
			--charbuf.current;
#endif
		      }
		    else
		      break;
		  }
		char_buffer_add (&charbuf, c);
		if (width > 0)
		  --width;

		c = inchar ();
	      }

	  if (char_buffer_error (&charbuf))
	    {
	      __set_errno (ENOMEM);
	      done = EOF;
	      goto errout;
	    }

	  if (char_buffer_size (&charbuf) == 0
	      || (char_buffer_size (&charbuf) == 1
		  && (char_buffer_start (&charbuf)[0] == L_('+')
		      || char_buffer_start (&charbuf)[0] == L_('-'))))
	    {
	      /* There was no number.  If we are supposed to read a pointer
		 we must recognize "(nil)" as well.  */
	      if (__builtin_expect (char_buffer_size (&charbuf) == 0
				    && (flags & READ_POINTER)
				    && (width < 0 || width >= 5)
				    && c == '('
				    && TOLOWER (inchar ()) == L_('n')
				    && TOLOWER (inchar ()) == L_('i')
				    && TOLOWER (inchar ()) == L_('l')
				    && inchar () == L_(')'), 1))
		/* We must produce the value of a NULL pointer.  A single
		   '0' digit is enough.  */
		  char_buffer_add (&charbuf, L_('0'));
	      else
		{
		  /* The last read character is not part of the number
		     anymore.  */
		  ungetc (c, s);

		  conv_error ();
		}
	    }
	  else
	    /* The just read character is not part of the number anymore.  */
	    ungetc (c, s);

	  /* Convert the number.  */
	  char_buffer_add (&charbuf, L_('\0'));
	  if (char_buffer_error (&charbuf))
	    {
	      __set_errno (ENOMEM);
	      done = EOF;
	      goto errout;
	    }
	  if (need_longlong && (flags & LONGDBL))
	    {
	      if (flags & NUMBER_SIGNED)
		num.q = __strtoll_internal
		  (char_buffer_start (&charbuf), &tw, base, flags & GROUP);
	      else
		num.uq = __strtoull_internal
		  (char_buffer_start (&charbuf), &tw, base, flags & GROUP);
	    }
	  else
	    {
	      if (flags & NUMBER_SIGNED)
		num.l = __strtol_internal
		  (char_buffer_start (&charbuf), &tw, base, flags & GROUP);
	      else
		num.ul = __strtoul_internal
		  (char_buffer_start (&charbuf), &tw, base, flags & GROUP);
	    }
	  if (__glibc_unlikely (char_buffer_start (&charbuf) == tw))
	    conv_error ();

	  if (!(flags & SUPPRESS))
	    {
	      if (flags & NUMBER_SIGNED)
		{
		  if (need_longlong && (flags & LONGDBL))
		    *ARG (LONGLONG int *) = num.q;
		  else if (need_long && (flags & LONG))
		    *ARG (long int *) = num.l;
		  else if (flags & SHORT)
		    *ARG (short int *) = (short int) num.l;
		  else if (!(flags & CHAR))
		    *ARG (int *) = (int) num.l;
		  else
		    *ARG (signed char *) = (signed char) num.ul;
		}
	      else
		{
		  if (need_longlong && (flags & LONGDBL))
		    *ARG (unsigned LONGLONG int *) = num.uq;
		  else if (need_long && (flags & LONG))
		    *ARG (unsigned long int *) = num.ul;
		  else if (flags & SHORT)
		    *ARG (unsigned short int *)
		      = (unsigned short int) num.ul;
		  else if (!(flags & CHAR))
		    *ARG (unsigned int *) = (unsigned int) num.ul;
		  else
		    *ARG (unsigned char *) = (unsigned char) num.ul;
		}
	      ++done;
	    }
	  break;

	case L_('e'):	/* Floating-point numbers.  */
	case L_('E'):
	case L_('f'):
	case L_('F'):
	case L_('g'):
	case L_('G'):
	case L_('a'):
	case L_('A'):
	  c = inchar ();
	  if (width > 0)
	    --width;
	  if (__glibc_unlikely (c == EOF))
	    input_error ();

	  got_digit = got_dot = got_e = got_sign = 0;

	  /* Check for a sign.  */
	  if (c == L_('-') || c == L_('+'))
	    {
	      got_sign = 1;
	      char_buffer_add (&charbuf, c);
	      if (__glibc_unlikely (width == 0 || inchar () == EOF))
		/* EOF is only an input error before we read any chars.  */
		conv_error ();
	      if (width > 0)
		--width;
	    }

	  /* Take care for the special arguments "nan" and "inf".  */
	  if (TOLOWER (c) == L_('n'))
	    {
	      /* Maybe "nan".  */
	      char_buffer_add (&charbuf, c);
	      if (__builtin_expect (width == 0
				    || inchar () == EOF
				    || TOLOWER (c) != L_('a'), 0))
		conv_error ();
	      if (width > 0)
		--width;
	      char_buffer_add (&charbuf, c);
	      if (__builtin_expect (width == 0
				    || inchar () == EOF
				    || TOLOWER (c) != L_('n'), 0))
		conv_error ();
	      if (width > 0)
		--width;
	      char_buffer_add (&charbuf, c);
	      /* It is "nan".  */
	      goto scan_float;
	    }
	  else if (TOLOWER (c) == L_('i'))
	    {
	      /* Maybe "inf" or "infinity".  */
	      char_buffer_add (&charbuf, c);
	      if (__builtin_expect (width == 0
				    || inchar () == EOF
				    || TOLOWER (c) != L_('n'), 0))
		conv_error ();
	      if (width > 0)
		--width;
	      char_buffer_add (&charbuf, c);
	      if (__builtin_expect (width == 0
				    || inchar () == EOF
				    || TOLOWER (c) != L_('f'), 0))
		conv_error ();
	      if (width > 0)
		--width;
	      char_buffer_add (&charbuf, c);
	      /* It is as least "inf".  */
	      if (width != 0 && inchar () != EOF)
		{
		  if (TOLOWER (c) == L_('i'))
		    {
		      if (width > 0)
			--width;
		      /* Now we have to read the rest as well.  */
		      char_buffer_add (&charbuf, c);
		      if (__builtin_expect (width == 0
					    || inchar () == EOF
					    || TOLOWER (c) != L_('n'), 0))
			conv_error ();
		      if (width > 0)
			--width;
		      char_buffer_add (&charbuf, c);
		      if (__builtin_expect (width == 0
					    || inchar () == EOF
					    || TOLOWER (c) != L_('i'), 0))
			conv_error ();
		      if (width > 0)
			--width;
		      char_buffer_add (&charbuf, c);
		      if (__builtin_expect (width == 0
					    || inchar () == EOF
					    || TOLOWER (c) != L_('t'), 0))
			conv_error ();
		      if (width > 0)
			--width;
		      char_buffer_add (&charbuf, c);
		      if (__builtin_expect (width == 0
					    || inchar () == EOF
					    || TOLOWER (c) != L_('y'), 0))
			conv_error ();
		      if (width > 0)
			--width;
		      char_buffer_add (&charbuf, c);
		    }
		  else
		    /* Never mind.  */
		    ungetc (c, s);
		}
	      goto scan_float;
	    }

	  exp_char = L_('e');
	  if (width != 0 && c == L_('0'))
	    {
	      char_buffer_add (&charbuf, c);
	      c = inchar ();
	      if (width > 0)
		--width;
	      if (width != 0 && TOLOWER (c) == L_('x'))
		{
		  /* It is a number in hexadecimal format.  */
		  char_buffer_add (&charbuf, c);

		  flags |= HEXA_FLOAT;
		  exp_char = L_('p');

		  /* Grouping is not allowed.  */
		  flags &= ~GROUP;
		  c = inchar ();
		  if (width > 0)
		    --width;
		}
	      else
		got_digit = 1;
	    }

	  while (1)
	    {
	      if (char_buffer_error (&charbuf))
		{
		  __set_errno (ENOMEM);
		  done = EOF;
		  goto errout;
		}
	      if (ISDIGIT (c))
		{
		  char_buffer_add (&charbuf, c);
		  got_digit = 1;
		}
	      else if (!got_e && (flags & HEXA_FLOAT) && ISXDIGIT (c))
		{
		  char_buffer_add (&charbuf, c);
		  got_digit = 1;
		}
	      else if (got_e && charbuf.current[-1] == exp_char
		       && (c == L_('-') || c == L_('+')))
		char_buffer_add (&charbuf, c);
	      else if (got_digit && !got_e
		       && (CHAR_T) TOLOWER (c) == exp_char)
		{
		  char_buffer_add (&charbuf, exp_char);
		  got_e = got_dot = 1;
		}
	      else
		{
#ifdef COMPILE_WSCANF
		  if (! got_dot && c == decimal)
		    {
		      char_buffer_add (&charbuf, c);
		      got_dot = 1;
		    }
		  else if ((flags & GROUP) != 0 && ! got_dot && c == thousands)
		    char_buffer_add (&charbuf, c);
		  else
		    {
		      /* The last read character is not part of the number
			 anymore.  */
		      ungetc (c, s);
		      break;
		    }
#else
		  const char *cmpp = decimal;
		  int avail = width > 0 ? width : INT_MAX;

		  if (! got_dot)
		    {
		      while ((unsigned char) *cmpp == c && avail >= 0)
			if (*++cmpp == '\0')
			  break;
			else
			  {
			    if (avail == 0 || inchar () == EOF)
			      break;
			    --avail;
			  }
		    }

		  if (*cmpp == '\0')
		    {
		      /* Add all the characters.  */
		      for (cmpp = decimal; *cmpp != '\0'; ++cmpp)
			char_buffer_add (&charbuf, (unsigned char) *cmpp);
		      if (width > 0)
			width = avail;
		      got_dot = 1;
		    }
		  else
		    {
		      /* Figure out whether it is a thousands separator.
			 There is one problem: we possibly read more than
			 one character.  We cannot push them back but since
			 we know that parts of the `decimal' string matched,
			 we can compare against it.  */
		      const char *cmp2p = thousands;

		      if ((flags & GROUP) != 0 && ! got_dot)
			{
			  while (cmp2p - thousands < cmpp - decimal
				 && *cmp2p == decimal[cmp2p - thousands])
			    ++cmp2p;
			  if (cmp2p - thousands == cmpp - decimal)
			    {
			      while ((unsigned char) *cmp2p == c && avail >= 0)
				if (*++cmp2p == '\0')
				  break;
				else
				  {
				    if (avail == 0 || inchar () == EOF)
				      break;
				    --avail;
				  }
			    }
			}

		      if (cmp2p != NULL && *cmp2p == '\0')
			{
			  /* Add all the characters.  */
			  for (cmpp = thousands; *cmpp != '\0'; ++cmpp)
			    char_buffer_add (&charbuf, (unsigned char) *cmpp);
			  if (width > 0)
			    width = avail;
			}
		      else
			{
			  /* The last read character is not part of the number
			     anymore.  */
			  ungetc (c, s);
			  break;
			}
		    }
#endif
		}

	      if (width == 0 || inchar () == EOF)
		break;

	      if (width > 0)
		--width;
	    }

	  if (char_buffer_error (&charbuf))
	    {
	      __set_errno (ENOMEM);
	      done = EOF;
	      goto errout;
	    }

	  wctrans_t map;
	  if (__builtin_expect ((flags & I18N) != 0, 0)
	      /* Hexadecimal floats make no sense, fixing localized
		 digits with ASCII letters.  */
	      && !(flags & HEXA_FLOAT)
	      /* Minimum requirement.  */
	      && (char_buffer_size (&charbuf) == got_sign || got_dot)
	      && (map = __wctrans ("to_inpunct")) != NULL)
	    {
	      /* Reget the first character.  */
	      inchar ();

	      /* Localized digits, decimal points, and thousands
		 separator.  */
	      wint_t wcdigits[12];

	      /* First get decimal equivalent to check if we read it
		 or not.  */
	      wcdigits[11] = __towctrans (L'.', map);

	      /* If we have not read any character or have just read
		 locale decimal point which matches the decimal point
		 for localized FP numbers, then we may have localized
		 digits.  Note, we test GOT_DOT above.  */
#ifdef COMPILE_WSCANF
	      if (char_buffer_size (&charbuf) == got_sign
		  || (char_buffer_size (&charbuf) == got_sign + 1
		      && wcdigits[11] == decimal))
#else
	      char mbdigits[12][MB_LEN_MAX + 1];

	      mbstate_t state;
	      memset (&state, '\0', sizeof (state));

	      bool match_so_far = char_buffer_size (&charbuf) == got_sign;
	      size_t mblen = __wcrtomb (mbdigits[11], wcdigits[11], &state);
	      if (mblen != (size_t) -1)
		{
		  mbdigits[11][mblen] = '\0';
		  match_so_far |=
		    (char_buffer_size (&charbuf) == strlen (decimal) + got_sign
		     && strcmp (decimal, mbdigits[11]) == 0);
		}
	      else
		{
		  size_t decimal_len = strlen (decimal);
		  /* This should always be the case but the data comes
		     from a file.  */
		  if (decimal_len <= MB_LEN_MAX)
		    {
		      match_so_far |= (char_buffer_size (&charbuf)
				       == decimal_len + got_sign);
		      memcpy (mbdigits[11], decimal, decimal_len + 1);
		    }
		  else
		    match_so_far = false;
		}

	      if (match_so_far)
#endif
		{
		  bool have_locthousands = (flags & GROUP) != 0;

		  /* Now get the digits and the thousands-sep equivalents.  */
		  for (int n = 0; n < 11; ++n)
		    {
		      if (n < 10)
			wcdigits[n] = __towctrans (L'0' + n, map);
		      else if (n == 10)
			{
			  wcdigits[10] = __towctrans (L',', map);
			  have_locthousands &= wcdigits[10] != L'\0';
			}

#ifndef COMPILE_WSCANF
		      memset (&state, '\0', sizeof (state));

		      size_t mblen = __wcrtomb (mbdigits[n], wcdigits[n],
						&state);
		      if (mblen == (size_t) -1)
			{
			  if (n == 10)
			    {
			      if (have_locthousands)
				{
				  size_t thousands_len = strlen (thousands);
				  if (thousands_len <= MB_LEN_MAX)
				    memcpy (mbdigits[10], thousands,
					    thousands_len + 1);
				  else
				    have_locthousands = false;
				}
			    }
			  else
			    /* Ignore checking against localized digits.  */
			    goto no_i18nflt;
			}
		      else
			mbdigits[n][mblen] = '\0';
#endif
		    }

		  /* Start checking against localized digits, if
		     conversion is done correctly. */
		  while (1)
		    {
		      if (char_buffer_error (&charbuf))
			{
			  __set_errno (ENOMEM);
			  done = EOF;
			  goto errout;
			}
		      if (got_e && charbuf.current[-1] == exp_char
			  && (c == L_('-') || c == L_('+')))
			char_buffer_add (&charbuf, c);
		      else if (char_buffer_size (&charbuf) > got_sign && !got_e
			       && (CHAR_T) TOLOWER (c) == exp_char)
			{
			  char_buffer_add (&charbuf, exp_char);
			  got_e = got_dot = 1;
			}
		      else
			{
			  /* Check against localized digits, decimal point,
			     and thousands separator.  */
			  int n;
			  for (n = 0; n < 12; ++n)
			    {
#ifdef COMPILE_WSCANF
			      if (c == wcdigits[n])
				{
				  if (n < 10)
				    char_buffer_add (&charbuf, L_('0') + n);
				  else if (n == 11 && !got_dot)
				    {
				      char_buffer_add (&charbuf, decimal);
				      got_dot = 1;
				    }
				  else if (n == 10 && have_locthousands
					   && ! got_dot)
				    char_buffer_add (&charbuf, thousands);
				  else
				    /* The last read character is not part
				       of the number anymore.  */
				    n = 12;

				  break;
				}
#else
			      const char *cmpp = mbdigits[n];
			      int avail = width > 0 ? width : INT_MAX;

			      while ((unsigned char) *cmpp == c && avail >= 0)
				if (*++cmpp == '\0')
				  break;
				else
				  {
				    if (avail == 0 || inchar () == EOF)
				      break;
				    --avail;
				  }
			      if (*cmpp == '\0')
				{
				  if (width > 0)
				    width = avail;

				  if (n < 10)
				    char_buffer_add (&charbuf, L_('0') + n);
				  else if (n == 11 && !got_dot)
				    {
				      /* Add all the characters.  */
				      for (cmpp = decimal; *cmpp != '\0';
					   ++cmpp)
					char_buffer_add (&charbuf,
							 (unsigned char) *cmpp);

				      got_dot = 1;
				    }
				  else if (n == 10 && (flags & GROUP) != 0
					   && ! got_dot)
				    {
				      /* Add all the characters.  */
				      for (cmpp = thousands; *cmpp != '\0';
					   ++cmpp)
					char_buffer_add (&charbuf,
							 (unsigned char) *cmpp);
				    }
				  else
				    /* The last read character is not part
				       of the number anymore.  */
				      n = 12;

				  break;
				}

			      /* We are pushing all read characters back.  */
			      if (cmpp > mbdigits[n])
				{
				  ungetc (c, s);
				  while (--cmpp > mbdigits[n])
				    ungetc_not_eof ((unsigned char) *cmpp, s);
				  c = (unsigned char) *cmpp;
				}
#endif
			    }

			  if (n >= 12)
			    {
			      /* The last read character is not part
				 of the number anymore.  */
			      ungetc (c, s);
			      break;
			    }
			}

		      if (width == 0 || inchar () == EOF)
			break;

		      if (width > 0)
			--width;
		    }
		}

#ifndef COMPILE_WSCANF
	    no_i18nflt:
	      ;
#endif
	    }

	  if (char_buffer_error (&charbuf))
	    {
	      __set_errno (ENOMEM);
	      done = EOF;
	      goto errout;
	    }

	  /* Have we read any character?  If we try to read a number
	     in hexadecimal notation and we have read only the `0x'
	     prefix this is an error.  */
	  if (__glibc_unlikely (char_buffer_size (&charbuf) == got_sign
				|| ((flags & HEXA_FLOAT)
				    && (char_buffer_size (&charbuf)
					== 2 + got_sign))))
	    conv_error ();

	scan_float:
	  /* Convert the number.  */
	  char_buffer_add (&charbuf, L_('\0'));
	  if (char_buffer_error (&charbuf))
	    {
	      __set_errno (ENOMEM);
	      done = EOF;
	      goto errout;
	    }
#if __HAVE_FLOAT128_UNLIKE_LDBL
	  if ((flags & LONGDBL) \
	       && (mode_flags & SCANF_LDBL_USES_FLOAT128) != 0)
	    {
	      _Float128 d = __strtof128_internal
		(char_buffer_start (&charbuf), &tw, flags & GROUP);
	      if (!(flags & SUPPRESS) && tw != char_buffer_start (&charbuf))
		*ARG (_Float128 *) = d;
	    }
	  else
#endif
	  if ((flags & LONGDBL) \
	      && __glibc_likely ((mode_flags & SCANF_LDBL_IS_DBL) == 0))
	    {
	      long double d = __strtold_internal
		(char_buffer_start (&charbuf), &tw, flags & GROUP);
	      if (!(flags & SUPPRESS) && tw != char_buffer_start (&charbuf))
		*ARG (long double *) = d;
	    }
	  else if (flags & (LONG | LONGDBL))
	    {
	      double d = __strtod_internal
		(char_buffer_start (&charbuf), &tw, flags & GROUP);
	      if (!(flags & SUPPRESS) && tw != char_buffer_start (&charbuf))
		*ARG (double *) = d;
	    }
	  else
	    {
	      float d = __strtof_internal
		(char_buffer_start (&charbuf), &tw, flags & GROUP);
	      if (!(flags & SUPPRESS) && tw != char_buffer_start (&charbuf))
		*ARG (float *) = d;
	    }

	  if (__glibc_unlikely (tw == char_buffer_start (&charbuf)))
	    conv_error ();

	  if (!(flags & SUPPRESS))
	    ++done;
	  break;

	case L_('['):	/* Character class.  */
	  if (flags & LONG)
	    STRING_ARG (wstr, wchar_t, 100);
	  else
	    STRING_ARG (str, char, 100);

	  if (*f == L_('^'))
	    {
	      ++f;
	      not_in = 1;
	    }
	  else
	    not_in = 0;


#ifdef COMPILE_WSCANF
	  /* Find the beginning and the end of the scanlist.  We are not
	     creating a lookup table since it would have to be too large.
	     Instead we search each time through the string.  This is not
	     a constant lookup time but who uses this feature deserves to
	     be punished.  */
	  tw = (wchar_t *) f;	/* Marks the beginning.  */

	  if (*f == L']')
	    ++f;

	  while ((fc = *f++) != L'\0' && fc != L']');

	  if (__glibc_unlikely (fc == L'\0'))
	    conv_error ();
	  wchar_t *twend = (wchar_t *) f - 1;
#else
	  /* Fill WP with byte flags indexed by character.
	     We will use this flag map for matching input characters.  */
	  if (!scratch_buffer_set_array_size
	      (&charbuf.scratch, UCHAR_MAX + 1, 1))
	    {
	      done = EOF;
	      goto errout;
	    }
	  memset (charbuf.scratch.data, '\0', UCHAR_MAX + 1);

	  fc = *f;
	  if (fc == ']' || fc == '-')
	    {
	      /* If ] or - appears before any char in the set, it is not
		 the terminator or separator, but the first char in the
		 set.  */
	      ((char *)charbuf.scratch.data)[fc] = 1;
	      ++f;
	    }

	  while ((fc = *f++) != '\0' && fc != ']')
	    if (fc == '-' && *f != '\0' && *f != ']' && f[-2] <= *f)
	      {
		/* Add all characters from the one before the '-'
		   up to (but not including) the next format char.  */
		for (fc = f[-2]; fc < *f; ++fc)
		  ((char *)charbuf.scratch.data)[fc] = 1;
	      }
	    else
	      /* Add the character to the flag map.  */
	      ((char *)charbuf.scratch.data)[fc] = 1;

	  if (__glibc_unlikely (fc == '\0'))
	    conv_error();
#endif

	  if (flags & LONG)
	    {
	      size_t now = read_in;
#ifdef COMPILE_WSCANF
	      if (__glibc_unlikely (inchar () == WEOF))
		input_error ();

	      do
		{
		  wchar_t *runp;

		  /* Test whether it's in the scanlist.  */
		  runp = tw;
		  while (runp < twend)
		    {
		      if (runp[0] == L'-' && runp[1] != '\0'
			  && runp + 1 != twend
			  && runp != tw
			  && (unsigned int) runp[-1] <= (unsigned int) runp[1])
			{
			  /* Match against all characters in between the
			     first and last character of the sequence.  */
			  wchar_t wc;

			  for (wc = runp[-1] + 1; wc <= runp[1]; ++wc)
			    if ((wint_t) wc == c)
			      break;

			  if (wc <= runp[1] && !not_in)
			    break;
			  if (wc <= runp[1] && not_in)
			    {
			      /* The current character is not in the
				 scanset.  */
			      ungetc (c, s);
			      goto out;
			    }

			  runp += 2;
			}
		      else
			{
			  if ((wint_t) *runp == c && !not_in)
			    break;
			  if ((wint_t) *runp == c && not_in)
			    {
			      ungetc (c, s);
			      goto out;
			    }

			  ++runp;
			}
		    }

		  if (runp == twend && !not_in)
		    {
		      ungetc (c, s);
		      goto out;
		    }

		  if (!(flags & SUPPRESS))
		    {
		      *wstr++ = c;

		      if ((flags & MALLOC)
			  && wstr == (wchar_t *) *strptr + strsize)
			{
			  /* Enlarge the buffer.  */
			  wstr = (wchar_t *) realloc (*strptr,
						      (2 * strsize)
						      * sizeof (wchar_t));
			  if (wstr == NULL)
			    {
			      /* Can't allocate that much.  Last-ditch
				 effort.  */
			      wstr = (wchar_t *)
				realloc (*strptr, (strsize + 1)
						  * sizeof (wchar_t));
			      if (wstr == NULL)
				{
				  if (flags & POSIX_MALLOC)
				    {
				      done = EOF;
				      goto errout;
				    }
				  /* We lose.  Oh well.  Terminate the string
				     and stop converting, so at least we don't
				     skip any input.  */
				  ((wchar_t *) (*strptr))[strsize - 1] = L'\0';
				  strptr = NULL;
				  ++done;
				  conv_error ();
				}
			      else
				{
				  *strptr = (char *) wstr;
				  wstr += strsize;
				  ++strsize;
				}
			    }
			  else
			    {
			      *strptr = (char *) wstr;
			      wstr += strsize;
			      strsize *= 2;
			    }
			}
		    }
		}
	      while ((width < 0 || --width > 0) && inchar () != WEOF);
	    out:
#else
	      char buf[MB_LEN_MAX];
	      size_t cnt = 0;
	      mbstate_t cstate;

	      if (__glibc_unlikely (inchar () == EOF))
		input_error ();

	      memset (&cstate, '\0', sizeof (cstate));

	      do
		{
		  if (((char *) charbuf.scratch.data)[c] == not_in)
		    {
		      ungetc_not_eof (c, s);
		      break;
		    }

		  /* This is easy.  */
		  if (!(flags & SUPPRESS))
		    {
		      size_t n;

		      /* Convert it into a wide character.  */
		      buf[0] = c;
		      n = __mbrtowc (wstr, buf, 1, &cstate);

		      if (n == (size_t) -2)
			{
			  /* Possibly correct character, just not enough
			     input.  */
			  ++cnt;
			  assert (cnt < MB_LEN_MAX);
			  continue;
			}
		      cnt = 0;

		      ++wstr;
		      if ((flags & MALLOC)
			  && wstr == (wchar_t *) *strptr + strsize)
			{
			  /* Enlarge the buffer.  */
			  wstr = (wchar_t *) realloc (*strptr,
						      (2 * strsize
						       * sizeof (wchar_t)));
			  if (wstr == NULL)
			    {
			      /* Can't allocate that much.  Last-ditch
				 effort.  */
			      wstr = (wchar_t *)
				realloc (*strptr, ((strsize + 1)
						   * sizeof (wchar_t)));
			      if (wstr == NULL)
				{
				  if (flags & POSIX_MALLOC)
				    {
				      done = EOF;
				      goto errout;
				    }
				  /* We lose.  Oh well.  Terminate the
				     string and stop converting,
				     so at least we don't skip any input.  */
				  ((wchar_t *) (*strptr))[strsize - 1] = L'\0';
				  strptr = NULL;
				  ++done;
				  conv_error ();
				}
			      else
				{
				  *strptr = (char *) wstr;
				  wstr += strsize;
				  ++strsize;
				}
			    }
			  else
			    {
			      *strptr = (char *) wstr;
			      wstr += strsize;
			      strsize *= 2;
			    }
			}
		    }

		  if (width >= 0 && --width <= 0)
		    break;
		}
	      while (inchar () != EOF);

	      if (__glibc_unlikely (cnt != 0))
		/* We stopped in the middle of recognizing another
		   character.  That's a problem.  */
		encode_error ();
#endif

	      if (__glibc_unlikely (now == read_in))
		/* We haven't succesfully read any character.  */
		conv_error ();

	      if (!(flags & SUPPRESS))
		{
		  *wstr++ = L'\0';

		  if ((flags & MALLOC)
		      && wstr - (wchar_t *) *strptr != strsize)
		    {
		      wchar_t *cp = (wchar_t *)
			realloc (*strptr, ((wstr - (wchar_t *) *strptr)
					   * sizeof (wchar_t)));
		      if (cp != NULL)
			*strptr = (char *) cp;
		    }
		  strptr = NULL;

		  ++done;
		}
	    }
	  else
	    {
	      size_t now = read_in;

	      if (__glibc_unlikely (inchar () == EOF))
		input_error ();

#ifdef COMPILE_WSCANF

	      memset (&state, '\0', sizeof (state));

	      do
		{
		  wchar_t *runp;
		  size_t n;

		  /* Test whether it's in the scanlist.  */
		  runp = tw;
		  while (runp < twend)
		    {
		      if (runp[0] == L'-' && runp[1] != '\0'
			  && runp + 1 != twend
			  && runp != tw
			  && (unsigned int) runp[-1] <= (unsigned int) runp[1])
			{
			  /* Match against all characters in between the
			     first and last character of the sequence.  */
			  wchar_t wc;

			  for (wc = runp[-1] + 1; wc <= runp[1]; ++wc)
			    if ((wint_t) wc == c)
			      break;

			  if (wc <= runp[1] && !not_in)
			    break;
			  if (wc <= runp[1] && not_in)
			    {
			      /* The current character is not in the
				 scanset.  */
			      ungetc (c, s);
			      goto out2;
			    }

			  runp += 2;
			}
		      else
			{
			  if ((wint_t) *runp == c && !not_in)
			    break;
			  if ((wint_t) *runp == c && not_in)
			    {
			      ungetc (c, s);
			      goto out2;
			    }

			  ++runp;
			}
		    }

		  if (runp == twend && !not_in)
		    {
		      ungetc (c, s);
		      goto out2;
		    }

		  if (!(flags & SUPPRESS))
		    {
		      if ((flags & MALLOC)
			  && *strptr + strsize - str <= MB_LEN_MAX)
			{
			  /* Enlarge the buffer.  */
			  size_t strleng = str - *strptr;
			  char *newstr;

			  newstr = (char *) realloc (*strptr, 2 * strsize);
			  if (newstr == NULL)
			    {
			      /* Can't allocate that much.  Last-ditch
				 effort.  */
			      newstr = (char *) realloc (*strptr,
							 strleng + MB_LEN_MAX);
			      if (newstr == NULL)
				{
				  if (flags & POSIX_MALLOC)
				    {
				      done = EOF;
				      goto errout;
				    }
				  /* We lose.  Oh well.  Terminate the string
				     and stop converting, so at least we don't
				     skip any input.  */
				  ((char *) (*strptr))[strleng] = '\0';
				  strptr = NULL;
				  ++done;
				  conv_error ();
				}
			      else
				{
				  *strptr = newstr;
				  str = newstr + strleng;
				  strsize = strleng + MB_LEN_MAX;
				}
			    }
			  else
			    {
			      *strptr = newstr;
			      str = newstr + strleng;
			      strsize *= 2;
			    }
			}
		    }

		  n = __wcrtomb (!(flags & SUPPRESS) ? str : NULL, c, &state);
		  if (__glibc_unlikely (n == (size_t) -1))
		    encode_error ();

		  assert (n <= MB_LEN_MAX);
		  str += n;
		}
	      while ((width < 0 || --width > 0) && inchar () != WEOF);
	    out2:
#else
	      do
		{
		  if (((char *) charbuf.scratch.data)[c] == not_in)
		    {
		      ungetc_not_eof (c, s);
		      break;
		    }

		  /* This is easy.  */
		  if (!(flags & SUPPRESS))
		    {
		      *str++ = c;
		      if ((flags & MALLOC)
			  && (char *) str == *strptr + strsize)
			{
			  /* Enlarge the buffer.  */
			  size_t newsize = 2 * strsize;

			allocagain:
			  str = (char *) realloc (*strptr, newsize);
			  if (str == NULL)
			    {
			      /* Can't allocate that much.  Last-ditch
				 effort.  */
			      if (newsize > strsize + 1)
				{
				  newsize = strsize + 1;
				  goto allocagain;
				}
			      if (flags & POSIX_MALLOC)
				{
				  done = EOF;
				  goto errout;
				}
			      /* We lose.  Oh well.  Terminate the
				 string and stop converting,
				 so at least we don't skip any input.  */
			      ((char *) (*strptr))[strsize - 1] = '\0';
			      strptr = NULL;
			      ++done;
			      conv_error ();
			    }
			  else
			    {
			      *strptr = (char *) str;
			      str += strsize;
			      strsize = newsize;
			    }
			}
		    }
		}
	      while ((width < 0 || --width > 0) && inchar () != EOF);
#endif

	      if (__glibc_unlikely (now == read_in))
		/* We haven't succesfully read any character.  */
		conv_error ();

	      if (!(flags & SUPPRESS))
		{
#ifdef COMPILE_WSCANF
		  /* We have to emit the code to get into the initial
		     state.  */
		  char buf[MB_LEN_MAX];
		  size_t n = __wcrtomb (buf, L'\0', &state);
		  if (n > 0 && (flags & MALLOC)
		      && str + n >= *strptr + strsize)
		    {
		      /* Enlarge the buffer.  */
		      size_t strleng = str - *strptr;
		      char *newstr;

		      newstr = (char *) realloc (*strptr, strleng + n + 1);
		      if (newstr == NULL)
			{
			  if (flags & POSIX_MALLOC)
			    {
			      done = EOF;
			      goto errout;
			    }
			  /* We lose.  Oh well.  Terminate the string
			     and stop converting, so at least we don't
			     skip any input.  */
			  ((char *) (*strptr))[strleng] = '\0';
			  strptr = NULL;
			  ++done;
			  conv_error ();
			}
		      else
			{
			  *strptr = newstr;
			  str = newstr + strleng;
			  strsize = strleng + n + 1;
			}
		    }

		  str = __mempcpy (str, buf, n);
#endif
		  *str++ = '\0';

		  if ((flags & MALLOC) && str - *strptr != strsize)
		    {
		      char *cp = (char *) realloc (*strptr, str - *strptr);
		      if (cp != NULL)
			*strptr = cp;
		    }
		  strptr = NULL;

		  ++done;
		}
	    }
	  break;

	case L_('p'):	/* Generic pointer.  */
	  base = 16;
	  /* A PTR must be the same size as a `long int'.  */
	  flags &= ~(SHORT|LONGDBL);
	  if (need_long)
	    flags |= LONG;
	  flags |= READ_POINTER;
	  goto number;

	default:
	  /* If this is an unknown format character punt.  */
	  conv_error ();
	}
    }

  /* The last thing we saw int the format string was a white space.
     Consume the last white spaces.  */
  if (skip_space)
    {
      do
	c = inchar ();
      while (ISSPACE (c));
      ungetc (c, s);
    }

 errout:
  /* Unlock stream.  */
  UNLOCK_STREAM (s);

  scratch_buffer_free (&charbuf.scratch);

  if (__glibc_unlikely (done == EOF))
    {
      if (__glibc_unlikely (ptrs_to_free != NULL))
	{
	  struct ptrs_to_free *p = ptrs_to_free;
	  while (p != NULL)
	    {
	      for (size_t cnt = 0; cnt < p->count; ++cnt)
		{
		  free (*p->ptrs[cnt]);
		  *p->ptrs[cnt] = NULL;
		}
	      p = p->next;
	      ptrs_to_free = p;
	    }
	}
    }
  else if (__glibc_unlikely (strptr != NULL))
    {
      free (*strptr);
      *strptr = NULL;
    }
  return done;
}
