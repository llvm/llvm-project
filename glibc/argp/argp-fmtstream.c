/* Word-wrapping and line-truncating streams
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
   don't have that.  */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdarg.h>
#include <ctype.h>

#include <argp-fmtstream.h>
#include "argp-namefrob.h"

#ifndef ARGP_FMTSTREAM_USE_LINEWRAP

#ifndef isblank
#define isblank(ch) ((ch)==' ' || (ch)=='\t')
#endif

#ifdef _LIBC
# include <wchar.h>
# include <libio/libioP.h>
#endif

#define INIT_BUF_SIZE 200
#define PRINTF_SIZE_GUESS 150

/* Return an argp_fmtstream that outputs to STREAM, and which prefixes lines
   written on it with LMARGIN spaces and limits them to RMARGIN columns
   total.  If WMARGIN >= 0, words that extend past RMARGIN are wrapped by
   replacing the whitespace before them with a newline and WMARGIN spaces.
   Otherwise, chars beyond RMARGIN are simply dropped until a newline.
   Returns NULL if there was an error.  */
argp_fmtstream_t
__argp_make_fmtstream (FILE *stream,
		       size_t lmargin, size_t rmargin, ssize_t wmargin)
{
  argp_fmtstream_t fs;

  fs = (struct argp_fmtstream *) malloc (sizeof (struct argp_fmtstream));
  if (fs != NULL)
    {
      fs->stream = stream;

      fs->lmargin = lmargin;
      fs->rmargin = rmargin;
      fs->wmargin = wmargin;
      fs->point_col = 0;
      fs->point_offs = 0;

      fs->buf = (char *) malloc (INIT_BUF_SIZE);
      if (! fs->buf)
	{
	  free (fs);
	  fs = 0;
	}
      else
	{
	  fs->p = fs->buf;
	  fs->end = fs->buf + INIT_BUF_SIZE;
	}
    }

  return fs;
}
#if 0
/* Not exported.  */
#ifdef weak_alias
weak_alias (__argp_make_fmtstream, argp_make_fmtstream)
#endif
#endif

/* Flush FS to its stream, and free it (but don't close the stream).  */
void
__argp_fmtstream_free (argp_fmtstream_t fs)
{
  __argp_fmtstream_update (fs);
  if (fs->p > fs->buf)
    {
#ifdef _LIBC
      __fxprintf (fs->stream, "%.*s", (int) (fs->p - fs->buf), fs->buf);
#else
      fwrite_unlocked (fs->buf, 1, fs->p - fs->buf, fs->stream);
#endif
    }
  free (fs->buf);
  free (fs);
}
#if 0
/* Not exported.  */
#ifdef weak_alias
weak_alias (__argp_fmtstream_free, argp_fmtstream_free)
#endif
#endif

/* Process FS's buffer so that line wrapping is done from POINT_OFFS to the
   end of its buffer.  This code is mostly from glibc stdio/linewrap.c.  */
void
__argp_fmtstream_update (argp_fmtstream_t fs)
{
  char *buf, *nl;
  size_t len;

  /* Scan the buffer for newlines.  */
  buf = fs->buf + fs->point_offs;
  while (buf < fs->p)
    {
      size_t r;

      if (fs->point_col == 0 && fs->lmargin != 0)
	{
	  /* We are starting a new line.  Print spaces to the left margin.  */
	  const size_t pad = fs->lmargin;
	  if (fs->p + pad < fs->end)
	    {
	      /* We can fit in them in the buffer by moving the
		 buffer text up and filling in the beginning.  */
	      memmove (buf + pad, buf, fs->p - buf);
	      fs->p += pad; /* Compensate for bigger buffer. */
	      memset (buf, ' ', pad); /* Fill in the spaces.  */
	      buf += pad; /* Don't bother searching them.  */
	    }
	  else
	    {
	      /* No buffer space for spaces.  Must flush.  */
	      size_t i;
	      for (i = 0; i < pad; i++)
		{
#ifdef _LIBC
		  if (_IO_fwide (fs->stream, 0) > 0)
		    putwc_unlocked (L' ', fs->stream);
		  else
#endif
		    putc_unlocked (' ', fs->stream);
		}
	    }
	  fs->point_col = pad;
	}

      len = fs->p - buf;
      nl = memchr (buf, '\n', len);

      if (fs->point_col < 0)
	fs->point_col = 0;

      if (!nl)
	{
	  /* The buffer ends in a partial line.  */

	  if (fs->point_col + len < fs->rmargin)
	    {
	      /* The remaining buffer text is a partial line and fits
		 within the maximum line width.  Advance point for the
		 characters to be written and stop scanning.  */
	      fs->point_col += len;
	      break;
	    }
	  else
	    /* Set the end-of-line pointer for the code below to
	       the end of the buffer.  */
	    nl = fs->p;
	}
      else if (fs->point_col + (nl - buf) < (ssize_t) fs->rmargin)
	{
	  /* The buffer contains a full line that fits within the maximum
	     line width.  Reset point and scan the next line.  */
	  fs->point_col = 0;
	  buf = nl + 1;
	  continue;
	}

      /* This line is too long.  */
      r = fs->rmargin - 1;

      if (fs->wmargin < 0)
	{
	  /* Truncate the line by overwriting the excess with the
	     newline and anything after it in the buffer.  */
	  if (nl < fs->p)
	    {
	      memmove (buf + (r - fs->point_col), nl, fs->p - nl);
	      fs->p -= buf + (r - fs->point_col) - nl;
	      /* Reset point for the next line and start scanning it.  */
	      fs->point_col = 0;
	      buf += r + 1; /* Skip full line plus \n. */
	    }
	  else
	    {
	      /* The buffer ends with a partial line that is beyond the
		 maximum line width.  Advance point for the characters
		 written, and discard those past the max from the buffer.  */
	      fs->point_col += len;
	      fs->p -= fs->point_col - r;
	      break;
	    }
	}
      else
	{
	  /* Do word wrap.  Go to the column just past the maximum line
	     width and scan back for the beginning of the word there.
	     Then insert a line break.  */

	  char *p, *nextline;
	  int i;

	  p = buf + (r + 1 - fs->point_col);
	  while (p >= buf && !isblank (*p))
	    --p;
	  nextline = p + 1;	/* This will begin the next line.  */

	  if (nextline > buf)
	    {
	      /* Swallow separating blanks.  */
	      if (p >= buf)
		do
		  --p;
		while (p >= buf && isblank (*p));
	      nl = p + 1;	/* The newline will replace the first blank. */
	    }
	  else
	    {
	      /* A single word that is greater than the maximum line width.
		 Oh well.  Put it on an overlong line by itself.  */
	      p = buf + (r + 1 - fs->point_col);
	      /* Find the end of the long word.  */
	      do
		++p;
	      while (p < nl && !isblank (*p));
	      if (p == nl)
		{
		  /* It already ends a line.  No fussing required.  */
		  fs->point_col = 0;
		  buf = nl + 1;
		  continue;
		}
	      /* We will move the newline to replace the first blank.  */
	      nl = p;
	      /* Swallow separating blanks.  */
	      do
		++p;
	      while (isblank (*p));
	      /* The next line will start here.  */
	      nextline = p;
	    }

	  /* Note: There are a bunch of tests below for
	     NEXTLINE == BUF + LEN + 1; this case is where NL happens to fall
	     at the end of the buffer, and NEXTLINE is in fact empty (and so
	     we need not be careful to maintain its contents).  */

	  if ((nextline == buf + len + 1
	       ? fs->end - nl < fs->wmargin + 1
	       : nextline - (nl + 1) < fs->wmargin)
	      && fs->p > nextline)
	    {
	      /* The margin needs more blanks than we removed.  */
	      if (fs->end - fs->p > fs->wmargin + 1)
		/* Make some space for them.  */
		{
		  size_t mv = fs->p - nextline;
		  memmove (nl + 1 + fs->wmargin, nextline, mv);
		  nextline = nl + 1 + fs->wmargin;
		  len = nextline + mv - buf;
		  *nl++ = '\n';
		}
	      else
		/* Output the first line so we can use the space.  */
		{
#ifdef _LIBC
		  __fxprintf (fs->stream, "%.*s\n",
			      (int) (nl - fs->buf), fs->buf);
#else
		  if (nl > fs->buf)
		    fwrite_unlocked (fs->buf, 1, nl - fs->buf, fs->stream);
		  putc_unlocked ('\n', fs->stream);
#endif

		  len += buf - fs->buf;
		  nl = buf = fs->buf;
		}
	    }
	  else
	    /* We can fit the newline and blanks in before
	       the next word.  */
	    *nl++ = '\n';

	  if (nextline - nl >= fs->wmargin
	      || (nextline == buf + len + 1 && fs->end - nextline >= fs->wmargin))
	    /* Add blanks up to the wrap margin column.  */
	    for (i = 0; i < fs->wmargin; ++i)
	      *nl++ = ' ';
	  else
	    for (i = 0; i < fs->wmargin; ++i)
#ifdef _LIBC
	      if (_IO_fwide (fs->stream, 0) > 0)
		putwc_unlocked (L' ', fs->stream);
	      else
#endif
		putc_unlocked (' ', fs->stream);

	  /* Copy the tail of the original buffer into the current buffer
	     position.  */
	  if (nl < nextline)
	    memmove (nl, nextline, buf + len - nextline);
	  len -= nextline - buf;

	  /* Continue the scan on the remaining lines in the buffer.  */
	  buf = nl;

	  /* Restore bufp to include all the remaining text.  */
	  fs->p = nl + len;

	  /* Reset the counter of what has been output this line.  If wmargin
	     is 0, we want to avoid the lmargin getting added, so we set
	     point_col to a magic value of -1 in that case.  */
	  fs->point_col = fs->wmargin ? fs->wmargin : -1;
	}
    }

  /* Remember that we've scanned as far as the end of the buffer.  */
  fs->point_offs = fs->p - fs->buf;
}

/* Ensure that FS has space for AMOUNT more bytes in its buffer, either by
   growing the buffer, or by flushing it.  True is returned iff we succeed. */
int
__argp_fmtstream_ensure (struct argp_fmtstream *fs, size_t amount)
{
  if ((size_t) (fs->end - fs->p) < amount)
    {
      ssize_t wrote;

      /* Flush FS's buffer.  */
      __argp_fmtstream_update (fs);

#ifdef _LIBC
      __fxprintf (fs->stream, "%.*s", (int) (fs->p - fs->buf), fs->buf);
      wrote = fs->p - fs->buf;
#else
      wrote = fwrite_unlocked (fs->buf, 1, fs->p - fs->buf, fs->stream);
#endif
      if (wrote == fs->p - fs->buf)
	{
	  fs->p = fs->buf;
	  fs->point_offs = 0;
	}
      else
	{
	  fs->p -= wrote;
	  fs->point_offs -= wrote;
	  memmove (fs->buf, fs->buf + wrote, fs->p - fs->buf);
	  return 0;
	}

      if ((size_t) (fs->end - fs->buf) < amount)
	/* Gotta grow the buffer.  */
	{
	  size_t old_size = fs->end - fs->buf;
	  size_t new_size = old_size + amount;
	  char *new_buf;

	  if (new_size < old_size || ! (new_buf = realloc (fs->buf, new_size)))
	    {
	      __set_errno (ENOMEM);
	      return 0;
	    }

	  fs->buf = new_buf;
	  fs->end = new_buf + new_size;
	  fs->p = fs->buf;
	}
    }

  return 1;
}

ssize_t
__argp_fmtstream_printf (struct argp_fmtstream *fs, const char *fmt, ...)
{
  int out;
  size_t avail;
  size_t size_guess = PRINTF_SIZE_GUESS; /* How much space to reserve. */

  do
    {
      va_list args;

      if (! __argp_fmtstream_ensure (fs, size_guess))
	return -1;

      va_start (args, fmt);
      avail = fs->end - fs->p;
      out = __vsnprintf_internal (fs->p, avail, fmt, args, 0);
      va_end (args);
      if ((size_t) out >= avail)
	size_guess = out + 1;
    }
  while ((size_t) out >= avail);

  fs->p += out;

  return out;
}
#if 0
/* Not exported.  */
#ifdef weak_alias
weak_alias (__argp_fmtstream_printf, argp_fmtstream_printf)
#endif
#endif

#endif /* !ARGP_FMTSTREAM_USE_LINEWRAP */
