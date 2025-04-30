/* Copyright (C) 1993-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Written by Ulrich Drepper <drepper@cygnus.com>.
   Based on the single byte version by Per Bothner <bothner@cygnus.com>.

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
   <https://www.gnu.org/licenses/>.

   As a special exception, if you link the code in this file with
   files compiled with a GNU compiler to produce an executable,
   that does not cause the resulting executable to be covered by
   the GNU Lesser General Public License.  This exception does not
   however invalidate any other reasons why the executable file
   might be covered by the GNU Lesser General Public License.
   This exception applies to code released by its copyright holders
   in files containing the exception.  */

/* Generic or default I/O operations. */

#include "libioP.h"
#include <stdlib.h>
#include <string.h>
#include <wchar.h>


static int save_for_wbackup (FILE *fp, wchar_t *end_p) __THROW;

/* Return minimum _pos markers
   Assumes the current get area is the main get area. */
ssize_t
_IO_least_wmarker (FILE *fp, wchar_t *end_p)
{
  ssize_t least_so_far = end_p - fp->_wide_data->_IO_read_base;
  struct _IO_marker *mark;
  for (mark = fp->_markers; mark != NULL; mark = mark->_next)
    if (mark->_pos < least_so_far)
      least_so_far = mark->_pos;
  return least_so_far;
}
libc_hidden_def (_IO_least_wmarker)

/* Switch current get area from backup buffer to (start of) main get area. */
void
_IO_switch_to_main_wget_area (FILE *fp)
{
  wchar_t *tmp;
  fp->_flags &= ~_IO_IN_BACKUP;
  /* Swap _IO_read_end and _IO_save_end. */
  tmp = fp->_wide_data->_IO_read_end;
  fp->_wide_data->_IO_read_end = fp->_wide_data->_IO_save_end;
  fp->_wide_data->_IO_save_end= tmp;
  /* Swap _IO_read_base and _IO_save_base. */
  tmp = fp->_wide_data->_IO_read_base;
  fp->_wide_data->_IO_read_base = fp->_wide_data->_IO_save_base;
  fp->_wide_data->_IO_save_base = tmp;
  /* Set _IO_read_ptr. */
  fp->_wide_data->_IO_read_ptr = fp->_wide_data->_IO_read_base;
}
libc_hidden_def (_IO_switch_to_main_wget_area)


/* Switch current get area from main get area to (end of) backup area. */
void
_IO_switch_to_wbackup_area (FILE *fp)
{
  wchar_t *tmp;
  fp->_flags |= _IO_IN_BACKUP;
  /* Swap _IO_read_end and _IO_save_end. */
  tmp = fp->_wide_data->_IO_read_end;
  fp->_wide_data->_IO_read_end = fp->_wide_data->_IO_save_end;
  fp->_wide_data->_IO_save_end = tmp;
  /* Swap _IO_read_base and _IO_save_base. */
  tmp = fp->_wide_data->_IO_read_base;
  fp->_wide_data->_IO_read_base = fp->_wide_data->_IO_save_base;
  fp->_wide_data->_IO_save_base = tmp;
  /* Set _IO_read_ptr.  */
  fp->_wide_data->_IO_read_ptr = fp->_wide_data->_IO_read_end;
}
libc_hidden_def (_IO_switch_to_wbackup_area)


void
_IO_wsetb (FILE *f, wchar_t *b, wchar_t *eb, int a)
{
  if (f->_wide_data->_IO_buf_base && !(f->_flags2 & _IO_FLAGS2_USER_WBUF))
    free (f->_wide_data->_IO_buf_base);
  f->_wide_data->_IO_buf_base = b;
  f->_wide_data->_IO_buf_end = eb;
  if (a)
    f->_flags2 &= ~_IO_FLAGS2_USER_WBUF;
  else
    f->_flags2 |= _IO_FLAGS2_USER_WBUF;
}
libc_hidden_def (_IO_wsetb)


wint_t
_IO_wdefault_pbackfail (FILE *fp, wint_t c)
{
  if (fp->_wide_data->_IO_read_ptr > fp->_wide_data->_IO_read_base
      && !_IO_in_backup (fp)
      && (wint_t) fp->_IO_read_ptr[-1] == c)
    --fp->_IO_read_ptr;
  else
    {
      /* Need to handle a filebuf in write mode (switch to read mode). FIXME!*/
      if (!_IO_in_backup (fp))
	{
	  /* We need to keep the invariant that the main get area
	     logically follows the backup area.  */
	  if (fp->_wide_data->_IO_read_ptr > fp->_wide_data->_IO_read_base
	      && _IO_have_wbackup (fp))
	    {
	      if (save_for_wbackup (fp, fp->_wide_data->_IO_read_ptr))
		return WEOF;
	    }
	  else if (!_IO_have_wbackup (fp))
	    {
	      /* No backup buffer: allocate one. */
	      /* Use nshort buffer, if unused? (probably not)  FIXME */
	      int backup_size = 128;
	      wchar_t *bbuf = (wchar_t *) malloc (backup_size
						  * sizeof (wchar_t));
	      if (bbuf == NULL)
		return WEOF;
	      fp->_wide_data->_IO_save_base = bbuf;
	      fp->_wide_data->_IO_save_end = (fp->_wide_data->_IO_save_base
					      + backup_size);
	      fp->_wide_data->_IO_backup_base = fp->_wide_data->_IO_save_end;
	    }
	  fp->_wide_data->_IO_read_base = fp->_wide_data->_IO_read_ptr;
	  _IO_switch_to_wbackup_area (fp);
	}
      else if (fp->_wide_data->_IO_read_ptr <= fp->_wide_data->_IO_read_base)
	{
	  /* Increase size of existing backup buffer. */
	  size_t new_size;
	  size_t old_size = (fp->_wide_data->_IO_read_end
                             - fp->_wide_data->_IO_read_base);
	  wchar_t *new_buf;
	  new_size = 2 * old_size;
	  new_buf = (wchar_t *) malloc (new_size * sizeof (wchar_t));
	  if (new_buf == NULL)
	    return WEOF;
	  __wmemcpy (new_buf + (new_size - old_size),
		     fp->_wide_data->_IO_read_base, old_size);
	  free (fp->_wide_data->_IO_read_base);
	  _IO_wsetg (fp, new_buf, new_buf + (new_size - old_size),
		     new_buf + new_size);
	  fp->_wide_data->_IO_backup_base = fp->_wide_data->_IO_read_ptr;
	}

      *--fp->_wide_data->_IO_read_ptr = c;
    }
  return c;
}
libc_hidden_def (_IO_wdefault_pbackfail)


void
_IO_wdefault_finish (FILE *fp, int dummy)
{
  struct _IO_marker *mark;
  if (fp->_wide_data->_IO_buf_base && !(fp->_flags2 & _IO_FLAGS2_USER_WBUF))
    {
      free (fp->_wide_data->_IO_buf_base);
      fp->_wide_data->_IO_buf_base = fp->_wide_data->_IO_buf_end = NULL;
    }

  for (mark = fp->_markers; mark != NULL; mark = mark->_next)
    mark->_sbuf = NULL;

  if (fp->_IO_save_base)
    {
      free (fp->_wide_data->_IO_save_base);
      fp->_IO_save_base = NULL;
    }

#ifdef _IO_MTSAFE_IO
  if (fp->_lock != NULL)
    _IO_lock_fini (*fp->_lock);
#endif

  _IO_un_link ((struct _IO_FILE_plus *) fp);
}
libc_hidden_def (_IO_wdefault_finish)


wint_t
_IO_wdefault_uflow (FILE *fp)
{
  wint_t wch;
  wch = _IO_UNDERFLOW (fp);
  if (wch == WEOF)
    return WEOF;
  return *fp->_wide_data->_IO_read_ptr++;
}
libc_hidden_def (_IO_wdefault_uflow)


wint_t
__woverflow (FILE *f, wint_t wch)
{
  if (f->_mode == 0)
    _IO_fwide (f, 1);
  return _IO_OVERFLOW (f, wch);
}
libc_hidden_def (__woverflow)


wint_t
__wuflow (FILE *fp)
{
  if (fp->_mode < 0 || (fp->_mode == 0 && _IO_fwide (fp, 1) != 1))
    return WEOF;

  if (fp->_mode == 0)
    _IO_fwide (fp, 1);
  if (_IO_in_put_mode (fp))
    if (_IO_switch_to_wget_mode (fp) == EOF)
      return WEOF;
  if (fp->_wide_data->_IO_read_ptr < fp->_wide_data->_IO_read_end)
    return *fp->_wide_data->_IO_read_ptr++;
  if (_IO_in_backup (fp))
    {
      _IO_switch_to_main_wget_area (fp);
      if (fp->_wide_data->_IO_read_ptr < fp->_wide_data->_IO_read_end)
	return *fp->_wide_data->_IO_read_ptr++;
    }
  if (_IO_have_markers (fp))
    {
      if (save_for_wbackup (fp, fp->_wide_data->_IO_read_end))
	return WEOF;
    }
  else if (_IO_have_wbackup (fp))
    _IO_free_wbackup_area (fp);
  return _IO_UFLOW (fp);
}
libc_hidden_def (__wuflow)

wint_t
__wunderflow (FILE *fp)
{
  if (fp->_mode < 0 || (fp->_mode == 0 && _IO_fwide (fp, 1) != 1))
    return WEOF;

  if (fp->_mode == 0)
    _IO_fwide (fp, 1);
  if (_IO_in_put_mode (fp))
    if (_IO_switch_to_wget_mode (fp) == EOF)
      return WEOF;
  if (fp->_wide_data->_IO_read_ptr < fp->_wide_data->_IO_read_end)
    return *fp->_wide_data->_IO_read_ptr;
  if (_IO_in_backup (fp))
    {
      _IO_switch_to_main_wget_area (fp);
      if (fp->_wide_data->_IO_read_ptr < fp->_wide_data->_IO_read_end)
	return *fp->_wide_data->_IO_read_ptr;
    }
  if (_IO_have_markers (fp))
    {
      if (save_for_wbackup (fp, fp->_wide_data->_IO_read_end))
	return WEOF;
    }
  else if (_IO_have_backup (fp))
    _IO_free_wbackup_area (fp);
  return _IO_UNDERFLOW (fp);
}
libc_hidden_def (__wunderflow)


size_t
_IO_wdefault_xsputn (FILE *f, const void *data, size_t n)
{
  const wchar_t *s = (const wchar_t *) data;
  size_t more = n;
  if (more <= 0)
    return 0;
  for (;;)
    {
      /* Space available. */
      ssize_t count = (f->_wide_data->_IO_write_end
                       - f->_wide_data->_IO_write_ptr);
      if (count > 0)
	{
	  if ((size_t) count > more)
	    count = more;
	  if (count > 20)
	    {
	      f->_wide_data->_IO_write_ptr =
		__wmempcpy (f->_wide_data->_IO_write_ptr, s, count);
	      s += count;
            }
	  else if (count <= 0)
	    count = 0;
	  else
	    {
	      wchar_t *p = f->_wide_data->_IO_write_ptr;
	      ssize_t i;
	      for (i = count; --i >= 0; )
		*p++ = *s++;
	      f->_wide_data->_IO_write_ptr = p;
            }
	  more -= count;
        }
      if (more == 0 || __woverflow (f, *s++) == WEOF)
	break;
      more--;
    }
  return n - more;
}
libc_hidden_def (_IO_wdefault_xsputn)


size_t
_IO_wdefault_xsgetn (FILE *fp, void *data, size_t n)
{
  size_t more = n;
  wchar_t *s = (wchar_t*) data;
  for (;;)
    {
      /* Data available. */
      ssize_t count = (fp->_wide_data->_IO_read_end
                       - fp->_wide_data->_IO_read_ptr);
      if (count > 0)
	{
	  if ((size_t) count > more)
	    count = more;
	  if (count > 20)
	    {
	      s = __wmempcpy (s, fp->_wide_data->_IO_read_ptr, count);
	      fp->_wide_data->_IO_read_ptr += count;
	    }
	  else if (count <= 0)
	    count = 0;
	  else
	    {
	      wchar_t *p = fp->_wide_data->_IO_read_ptr;
	      int i = (int) count;
	      while (--i >= 0)
		*s++ = *p++;
	      fp->_wide_data->_IO_read_ptr = p;
            }
            more -= count;
        }
      if (more == 0 || __wunderflow (fp) == WEOF)
	break;
    }
  return n - more;
}
libc_hidden_def (_IO_wdefault_xsgetn)


void
_IO_wdoallocbuf (FILE *fp)
{
  if (fp->_wide_data->_IO_buf_base)
    return;
  if (!(fp->_flags & _IO_UNBUFFERED))
    if ((wint_t)_IO_WDOALLOCATE (fp) != WEOF)
      return;
  _IO_wsetb (fp, fp->_wide_data->_shortbuf,
		     fp->_wide_data->_shortbuf + 1, 0);
}
libc_hidden_def (_IO_wdoallocbuf)


int
_IO_wdefault_doallocate (FILE *fp)
{
  wchar_t *buf = (wchar_t *)malloc (BUFSIZ);
  if (__glibc_unlikely (buf == NULL))
    return EOF;

  _IO_wsetb (fp, buf, buf + BUFSIZ / sizeof *buf, 1);
  return 1;
}
libc_hidden_def (_IO_wdefault_doallocate)


int
_IO_switch_to_wget_mode (FILE *fp)
{
  if (fp->_wide_data->_IO_write_ptr > fp->_wide_data->_IO_write_base)
    if ((wint_t)_IO_WOVERFLOW (fp, WEOF) == WEOF)
      return EOF;
  if (_IO_in_backup (fp))
    fp->_wide_data->_IO_read_base = fp->_wide_data->_IO_backup_base;
  else
    {
      fp->_wide_data->_IO_read_base = fp->_wide_data->_IO_buf_base;
      if (fp->_wide_data->_IO_write_ptr > fp->_wide_data->_IO_read_end)
	fp->_wide_data->_IO_read_end = fp->_wide_data->_IO_write_ptr;
    }
  fp->_wide_data->_IO_read_ptr = fp->_wide_data->_IO_write_ptr;

  fp->_wide_data->_IO_write_base = fp->_wide_data->_IO_write_ptr
    = fp->_wide_data->_IO_write_end = fp->_wide_data->_IO_read_ptr;

  fp->_flags &= ~_IO_CURRENTLY_PUTTING;
  return 0;
}
libc_hidden_def (_IO_switch_to_wget_mode)

void
_IO_free_wbackup_area (FILE *fp)
{
  if (_IO_in_backup (fp))
    _IO_switch_to_main_wget_area (fp);  /* Just in case. */
  free (fp->_wide_data->_IO_save_base);
  fp->_wide_data->_IO_save_base = NULL;
  fp->_wide_data->_IO_save_end = NULL;
  fp->_wide_data->_IO_backup_base = NULL;
}
libc_hidden_def (_IO_free_wbackup_area)

static int
save_for_wbackup (FILE *fp, wchar_t *end_p)
{
  /* Append [_IO_read_base..end_p] to backup area. */
  ssize_t least_mark = _IO_least_wmarker (fp, end_p);
  /* needed_size is how much space we need in the backup area. */
  size_t needed_size = ((end_p - fp->_wide_data->_IO_read_base)
                        - least_mark);
  /* FIXME: Dubious arithmetic if pointers are NULL */
  size_t current_Bsize = (fp->_wide_data->_IO_save_end
                          - fp->_wide_data->_IO_save_base);
  size_t avail; /* Extra space available for future expansion. */
  ssize_t delta;
  struct _IO_marker *mark;
  if (needed_size > current_Bsize)
    {
      wchar_t *new_buffer;
      avail = 100;
      new_buffer = (wchar_t *) malloc ((avail + needed_size)
				       * sizeof (wchar_t));
      if (new_buffer == NULL)
	return EOF;		/* FIXME */
      if (least_mark < 0)
	{
	  __wmempcpy (__wmempcpy (new_buffer + avail,
				  fp->_wide_data->_IO_save_end + least_mark,
				  -least_mark),
		      fp->_wide_data->_IO_read_base,
		      end_p - fp->_wide_data->_IO_read_base);
	}
      else
	{
	  __wmemcpy (new_buffer + avail,
		     fp->_wide_data->_IO_read_base + least_mark,
		     needed_size);
	}
      free (fp->_wide_data->_IO_save_base);
      fp->_wide_data->_IO_save_base = new_buffer;
      fp->_wide_data->_IO_save_end = new_buffer + avail + needed_size;
    }
  else
    {
      avail = current_Bsize - needed_size;
      if (least_mark < 0)
	{
	  __wmemmove (fp->_wide_data->_IO_save_base + avail,
		      fp->_wide_data->_IO_save_end + least_mark,
		      -least_mark);
	  __wmemcpy (fp->_wide_data->_IO_save_base + avail - least_mark,
		     fp->_wide_data->_IO_read_base,
		     end_p - fp->_wide_data->_IO_read_base);
	}
      else if (needed_size > 0)
	__wmemcpy (fp->_wide_data->_IO_save_base + avail,
		   fp->_wide_data->_IO_read_base + least_mark,
		   needed_size);
    }
  fp->_wide_data->_IO_backup_base = fp->_wide_data->_IO_save_base + avail;
  /* Adjust all the streammarkers. */
  delta = end_p - fp->_wide_data->_IO_read_base;
  for (mark = fp->_markers; mark != NULL; mark = mark->_next)
    mark->_pos -= delta;
  return 0;
}

wint_t
_IO_sputbackwc (FILE *fp, wint_t c)
{
  wint_t result;

  if (fp->_wide_data->_IO_read_ptr > fp->_wide_data->_IO_read_base
      && (wchar_t)fp->_wide_data->_IO_read_ptr[-1] == (wchar_t) c)
    {
      fp->_wide_data->_IO_read_ptr--;
      result = c;
    }
  else
    result = _IO_PBACKFAIL (fp, c);

  if (result != WEOF)
    fp->_flags &= ~_IO_EOF_SEEN;

  return result;
}
libc_hidden_def (_IO_sputbackwc)

wint_t
_IO_sungetwc (FILE *fp)
{
  wint_t result;

  if (fp->_wide_data->_IO_read_ptr > fp->_wide_data->_IO_read_base)
    {
      fp->_wide_data->_IO_read_ptr--;
      result = *fp->_wide_data->_IO_read_ptr;
    }
  else
    result = _IO_PBACKFAIL (fp, EOF);

  if (result != WEOF)
    fp->_flags &= ~_IO_EOF_SEEN;

  return result;
}


unsigned
_IO_adjust_wcolumn (unsigned start, const wchar_t *line, int count)
{
  const wchar_t *ptr = line + count;
  while (ptr > line)
    if (*--ptr == L'\n')
      return line + count - ptr - 1;
  return start + count;
}

void
_IO_init_wmarker (struct _IO_marker *marker, FILE *fp)
{
  marker->_sbuf = fp;
  if (_IO_in_put_mode (fp))
    _IO_switch_to_wget_mode (fp);
  if (_IO_in_backup (fp))
    marker->_pos = fp->_wide_data->_IO_read_ptr - fp->_wide_data->_IO_read_end;
  else
    marker->_pos = (fp->_wide_data->_IO_read_ptr
		    - fp->_wide_data->_IO_read_base);

  /* Should perhaps sort the chain? */
  marker->_next = fp->_markers;
  fp->_markers = marker;
}

#define BAD_DELTA EOF

/* Return difference between MARK and current position of MARK's stream. */
int
_IO_wmarker_delta (struct _IO_marker *mark)
{
  int cur_pos;
  if (mark->_sbuf == NULL)
    return BAD_DELTA;
  if (_IO_in_backup (mark->_sbuf))
    cur_pos = (mark->_sbuf->_wide_data->_IO_read_ptr
	       - mark->_sbuf->_wide_data->_IO_read_end);
  else
    cur_pos = (mark->_sbuf->_wide_data->_IO_read_ptr
	       - mark->_sbuf->_wide_data->_IO_read_base);
  return mark->_pos - cur_pos;
}

int
_IO_seekwmark (FILE *fp, struct _IO_marker *mark, int delta)
{
  if (mark->_sbuf != fp)
    return EOF;
 if (mark->_pos >= 0)
    {
      if (_IO_in_backup (fp))
	_IO_switch_to_main_wget_area (fp);
      fp->_wide_data->_IO_read_ptr = (fp->_wide_data->_IO_read_base
				      + mark->_pos);
    }
  else
    {
      if (!_IO_in_backup (fp))
	_IO_switch_to_wbackup_area (fp);
      fp->_wide_data->_IO_read_ptr = fp->_wide_data->_IO_read_end + mark->_pos;
    }
  return 0;
}

void
_IO_unsave_wmarkers (FILE *fp)
{
  struct _IO_marker *mark = fp->_markers;
  if (mark)
    {
      fp->_markers = 0;
    }

  if (_IO_have_backup (fp))
    _IO_free_wbackup_area (fp);
}
