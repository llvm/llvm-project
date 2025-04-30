/* Word-wrapping and line-truncating streams.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Written by Miles Bader <miles@gnu.ai.mit.edu>.

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

/* This package emulates glibc `line_wrap_stream' semantics for systems that
   don't have that.  If the system does have it, it is just a wrapper for
   that.  This header file is only used internally while compiling argp, and
   shouldn't be installed.  */

#ifndef _ARGP_FMTSTREAM_H
#define _ARGP_FMTSTREAM_H

#include <stdio.h>
#include <string.h>
#include <unistd.h>

#if defined (__GNU_LIBRARY__) && defined (HAVE_LINEWRAP_H)
/* line_wrap_stream is available, so use that.  */
#define ARGP_FMTSTREAM_USE_LINEWRAP
#endif

#ifdef ARGP_FMTSTREAM_USE_LINEWRAP
/* Just be a simple wrapper for line_wrap_stream; the semantics are
   *slightly* different, as line_wrap_stream doesn't actually make a new
   object, it just modifies the given stream (reversibly) to do
   line-wrapping.  Since we control who uses this code, it doesn't matter.  */

#include <linewrap.h>

typedef FILE *argp_fmtstream_t;

#define argp_make_fmtstream line_wrap_stream
#define __argp_make_fmtstream line_wrap_stream
#define argp_fmtstream_free line_unwrap_stream
#define __argp_fmtstream_free line_unwrap_stream

#define __argp_fmtstream_putc(fs,ch) putc(ch,fs)
#define argp_fmtstream_putc(fs,ch) putc(ch,fs)
#define __argp_fmtstream_puts(fs,str) fputs(str,fs)
#define argp_fmtstream_puts(fs,str) fputs(str,fs)
#define __argp_fmtstream_write(fs,str,len) fwrite(str,1,len,fs)
#define argp_fmtstream_write(fs,str,len) fwrite(str,1,len,fs)
#define __argp_fmtstream_printf fprintf
#define argp_fmtstream_printf fprintf

#define __argp_fmtstream_lmargin line_wrap_lmargin
#define argp_fmtstream_lmargin line_wrap_lmargin
#define __argp_fmtstream_set_lmargin line_wrap_set_lmargin
#define argp_fmtstream_set_lmargin line_wrap_set_lmargin
#define __argp_fmtstream_rmargin line_wrap_rmargin
#define argp_fmtstream_rmargin line_wrap_rmargin
#define __argp_fmtstream_set_rmargin line_wrap_set_rmargin
#define argp_fmtstream_set_rmargin line_wrap_set_rmargin
#define __argp_fmtstream_wmargin line_wrap_wmargin
#define argp_fmtstream_wmargin line_wrap_wmargin
#define __argp_fmtstream_set_wmargin line_wrap_set_wmargin
#define argp_fmtstream_set_wmargin line_wrap_set_wmargin
#define __argp_fmtstream_point line_wrap_point
#define argp_fmtstream_point line_wrap_point

#else /* !ARGP_FMTSTREAM_USE_LINEWRAP */
/* Guess we have to define our own version.  */


struct argp_fmtstream
{
  FILE *stream;			/* The stream we're outputting to.  */

  size_t lmargin, rmargin;	/* Left and right margins.  */
  ssize_t wmargin;		/* Margin to wrap to, or -1 to truncate.  */

  /* Point in buffer to which we've processed for wrapping, but not output.  */
  size_t point_offs;
  /* Output column at POINT_OFFS, or -1 meaning 0 but don't add lmargin.  */
  ssize_t point_col;

  char *buf;			/* Output buffer.  */
  char *p;			/* Current end of text in BUF. */
  char *end;			/* Absolute end of BUF.  */
};

typedef struct argp_fmtstream *argp_fmtstream_t;

__BEGIN_DECLS

/* Return an argp_fmtstream that outputs to STREAM, and which prefixes lines
   written on it with LMARGIN spaces and limits them to RMARGIN columns
   total.  If WMARGIN >= 0, words that extend past RMARGIN are wrapped by
   replacing the whitespace before them with a newline and WMARGIN spaces.
   Otherwise, chars beyond RMARGIN are simply dropped until a newline.
   Returns NULL if there was an error.  */
extern argp_fmtstream_t __argp_make_fmtstream (FILE *__stream,
					       size_t __lmargin,
					       size_t __rmargin,
					       ssize_t __wmargin);
extern argp_fmtstream_t argp_make_fmtstream (FILE *__stream,
					     size_t __lmargin,
					     size_t __rmargin,
					     ssize_t __wmargin);

/* Flush __FS to its stream, and free it (but don't close the stream).  */
extern void __argp_fmtstream_free (argp_fmtstream_t __fs);
extern void argp_fmtstream_free (argp_fmtstream_t __fs);

extern ssize_t __argp_fmtstream_printf (argp_fmtstream_t __fs,
					const char *__fmt, ...)
     __attribute__ ((__format__ (printf, 2, 3)));
extern ssize_t argp_fmtstream_printf (argp_fmtstream_t __fs,
				      const char *__fmt, ...)
     __attribute__ ((__format__ (printf, 2, 3)));

extern int __argp_fmtstream_putc (argp_fmtstream_t __fs, int __ch);
extern int argp_fmtstream_putc (argp_fmtstream_t __fs, int __ch);

extern int __argp_fmtstream_puts (argp_fmtstream_t __fs, const char *__str);
extern int argp_fmtstream_puts (argp_fmtstream_t __fs, const char *__str);

extern size_t __argp_fmtstream_write (argp_fmtstream_t __fs,
				      const char *__str, size_t __len);
extern size_t argp_fmtstream_write (argp_fmtstream_t __fs,
				    const char *__str, size_t __len);

/* Access macros for various bits of state.  */
#define argp_fmtstream_lmargin(__fs) ((__fs)->lmargin)
#define argp_fmtstream_rmargin(__fs) ((__fs)->rmargin)
#define argp_fmtstream_wmargin(__fs) ((__fs)->wmargin)
#define __argp_fmtstream_lmargin argp_fmtstream_lmargin
#define __argp_fmtstream_rmargin argp_fmtstream_rmargin
#define __argp_fmtstream_wmargin argp_fmtstream_wmargin

/* Set __FS's left margin to LMARGIN and return the old value.  */
extern size_t argp_fmtstream_set_lmargin (argp_fmtstream_t __fs,
					  size_t __lmargin);
extern size_t __argp_fmtstream_set_lmargin (argp_fmtstream_t __fs,
					    size_t __lmargin);

/* Set __FS's right margin to __RMARGIN and return the old value.  */
extern size_t argp_fmtstream_set_rmargin (argp_fmtstream_t __fs,
					  size_t __rmargin);
extern size_t __argp_fmtstream_set_rmargin (argp_fmtstream_t __fs,
					    size_t __rmargin);

/* Set __FS's wrap margin to __WMARGIN and return the old value.  */
extern size_t argp_fmtstream_set_wmargin (argp_fmtstream_t __fs,
					  size_t __wmargin);
extern size_t __argp_fmtstream_set_wmargin (argp_fmtstream_t __fs,
					    size_t __wmargin);

/* Return the column number of the current output point in __FS.  */
extern size_t argp_fmtstream_point (argp_fmtstream_t __fs);
extern size_t __argp_fmtstream_point (argp_fmtstream_t __fs);

/* Internal routines.  */
extern void _argp_fmtstream_update (argp_fmtstream_t __fs);
extern void __argp_fmtstream_update (argp_fmtstream_t __fs);
extern int _argp_fmtstream_ensure (argp_fmtstream_t __fs, size_t __amount);
extern int __argp_fmtstream_ensure (argp_fmtstream_t __fs, size_t __amount);

#ifdef __OPTIMIZE__
/* Inline versions of above routines.  */

#if !_LIBC
#define __argp_fmtstream_putc argp_fmtstream_putc
#define __argp_fmtstream_puts argp_fmtstream_puts
#define __argp_fmtstream_write argp_fmtstream_write
#define __argp_fmtstream_set_lmargin argp_fmtstream_set_lmargin
#define __argp_fmtstream_set_rmargin argp_fmtstream_set_rmargin
#define __argp_fmtstream_set_wmargin argp_fmtstream_set_wmargin
#define __argp_fmtstream_point argp_fmtstream_point
#define __argp_fmtstream_update _argp_fmtstream_update
#define __argp_fmtstream_ensure _argp_fmtstream_ensure
#endif

#ifndef ARGP_FS_EI
#define ARGP_FS_EI extern inline
#endif

ARGP_FS_EI size_t
__argp_fmtstream_write (argp_fmtstream_t __fs, const char *__str, size_t __len)
{
  if (__fs->p + __len <= __fs->end || __argp_fmtstream_ensure (__fs, __len))
    {
      memcpy (__fs->p, __str, __len);
      __fs->p += __len;
      return __len;
    }
  else
    return 0;
}

ARGP_FS_EI int
__argp_fmtstream_puts (argp_fmtstream_t __fs, const char *__str)
{
  size_t __len = strlen (__str);
  if (__len)
    {
      size_t __wrote = __argp_fmtstream_write (__fs, __str, __len);
      return __wrote == __len ? 0 : -1;
    }
  else
    return 0;
}

ARGP_FS_EI int
__argp_fmtstream_putc (argp_fmtstream_t __fs, int __ch)
{
  if (__fs->p < __fs->end || __argp_fmtstream_ensure (__fs, 1))
    return *__fs->p++ = __ch;
  else
    return EOF;
}

/* Set __FS's left margin to __LMARGIN and return the old value.  */
ARGP_FS_EI size_t
__argp_fmtstream_set_lmargin (argp_fmtstream_t __fs, size_t __lmargin)
{
  size_t __old;
  if ((size_t) (__fs->p - __fs->buf) > __fs->point_offs)
    __argp_fmtstream_update (__fs);
  __old = __fs->lmargin;
  __fs->lmargin = __lmargin;
  return __old;
}

/* Set __FS's right margin to __RMARGIN and return the old value.  */
ARGP_FS_EI size_t
__argp_fmtstream_set_rmargin (argp_fmtstream_t __fs, size_t __rmargin)
{
  size_t __old;
  if ((size_t) (__fs->p - __fs->buf) > __fs->point_offs)
    __argp_fmtstream_update (__fs);
  __old = __fs->rmargin;
  __fs->rmargin = __rmargin;
  return __old;
}

/* Set FS's wrap margin to __WMARGIN and return the old value.  */
ARGP_FS_EI size_t
__argp_fmtstream_set_wmargin (argp_fmtstream_t __fs, size_t __wmargin)
{
  size_t __old;
  if ((size_t) (__fs->p - __fs->buf) > __fs->point_offs)
    __argp_fmtstream_update (__fs);
  __old = __fs->wmargin;
  __fs->wmargin = __wmargin;
  return __old;
}

/* Return the column number of the current output point in __FS.  */
ARGP_FS_EI size_t
__argp_fmtstream_point (argp_fmtstream_t __fs)
{
  if ((size_t) (__fs->p - __fs->buf) > __fs->point_offs)
    __argp_fmtstream_update (__fs);
  return __fs->point_col >= 0 ? __fs->point_col : 0;
}

#if !_LIBC
#undef __argp_fmtstream_putc
#undef __argp_fmtstream_puts
#undef __argp_fmtstream_write
#undef __argp_fmtstream_set_lmargin
#undef __argp_fmtstream_set_rmargin
#undef __argp_fmtstream_set_wmargin
#undef __argp_fmtstream_point
#undef __argp_fmtstream_update
#undef __argp_fmtstream_ensure
#endif

#endif /* __OPTIMIZE__ */

__END_DECLS

#endif /* ARGP_FMTSTREAM_USE_LINEWRAP */

#endif /* argp-fmtstream.h */
