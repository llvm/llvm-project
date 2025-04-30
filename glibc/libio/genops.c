/* Copyright (C) 1993-2021 Free Software Foundation, Inc.
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
#include <stdbool.h>
#include <sched.h>

#ifdef _IO_MTSAFE_IO
static _IO_lock_t list_all_lock = _IO_lock_initializer;
#endif

static FILE *run_fp;

#ifdef _IO_MTSAFE_IO
static void
flush_cleanup (void *not_used)
{
  if (run_fp != NULL)
    _IO_funlockfile (run_fp);
  _IO_lock_unlock (list_all_lock);
}
#endif

void
_IO_un_link (struct _IO_FILE_plus *fp)
{
  if (fp->file._flags & _IO_LINKED)
    {
      FILE **f;
#ifdef _IO_MTSAFE_IO
      _IO_cleanup_region_start_noarg (flush_cleanup);
      _IO_lock_lock (list_all_lock);
      run_fp = (FILE *) fp;
      _IO_flockfile ((FILE *) fp);
#endif
      if (_IO_list_all == NULL)
	;
      else if (fp == _IO_list_all)
	_IO_list_all = (struct _IO_FILE_plus *) _IO_list_all->file._chain;
      else
	for (f = &_IO_list_all->file._chain; *f; f = &(*f)->_chain)
	  if (*f == (FILE *) fp)
	    {
	      *f = fp->file._chain;
	      break;
	    }
      fp->file._flags &= ~_IO_LINKED;
#ifdef _IO_MTSAFE_IO
      _IO_funlockfile ((FILE *) fp);
      run_fp = NULL;
      _IO_lock_unlock (list_all_lock);
      _IO_cleanup_region_end (0);
#endif
    }
}
libc_hidden_def (_IO_un_link)

void
_IO_link_in (struct _IO_FILE_plus *fp)
{
  if ((fp->file._flags & _IO_LINKED) == 0)
    {
      fp->file._flags |= _IO_LINKED;
#ifdef _IO_MTSAFE_IO
      _IO_cleanup_region_start_noarg (flush_cleanup);
      _IO_lock_lock (list_all_lock);
      run_fp = (FILE *) fp;
      _IO_flockfile ((FILE *) fp);
#endif
      fp->file._chain = (FILE *) _IO_list_all;
      _IO_list_all = fp;
#ifdef _IO_MTSAFE_IO
      _IO_funlockfile ((FILE *) fp);
      run_fp = NULL;
      _IO_lock_unlock (list_all_lock);
      _IO_cleanup_region_end (0);
#endif
    }
}
libc_hidden_def (_IO_link_in)

/* Return minimum _pos markers
   Assumes the current get area is the main get area. */
ssize_t _IO_least_marker (FILE *fp, char *end_p);

ssize_t
_IO_least_marker (FILE *fp, char *end_p)
{
  ssize_t least_so_far = end_p - fp->_IO_read_base;
  struct _IO_marker *mark;
  for (mark = fp->_markers; mark != NULL; mark = mark->_next)
    if (mark->_pos < least_so_far)
      least_so_far = mark->_pos;
  return least_so_far;
}

/* Switch current get area from backup buffer to (start of) main get area. */

void
_IO_switch_to_main_get_area (FILE *fp)
{
  char *tmp;
  fp->_flags &= ~_IO_IN_BACKUP;
  /* Swap _IO_read_end and _IO_save_end. */
  tmp = fp->_IO_read_end;
  fp->_IO_read_end = fp->_IO_save_end;
  fp->_IO_save_end= tmp;
  /* Swap _IO_read_base and _IO_save_base. */
  tmp = fp->_IO_read_base;
  fp->_IO_read_base = fp->_IO_save_base;
  fp->_IO_save_base = tmp;
  /* Set _IO_read_ptr. */
  fp->_IO_read_ptr = fp->_IO_read_base;
}

/* Switch current get area from main get area to (end of) backup area. */

void
_IO_switch_to_backup_area (FILE *fp)
{
  char *tmp;
  fp->_flags |= _IO_IN_BACKUP;
  /* Swap _IO_read_end and _IO_save_end. */
  tmp = fp->_IO_read_end;
  fp->_IO_read_end = fp->_IO_save_end;
  fp->_IO_save_end = tmp;
  /* Swap _IO_read_base and _IO_save_base. */
  tmp = fp->_IO_read_base;
  fp->_IO_read_base = fp->_IO_save_base;
  fp->_IO_save_base = tmp;
  /* Set _IO_read_ptr.  */
  fp->_IO_read_ptr = fp->_IO_read_end;
}

int
_IO_switch_to_get_mode (FILE *fp)
{
  if (fp->_IO_write_ptr > fp->_IO_write_base)
    if (_IO_OVERFLOW (fp, EOF) == EOF)
      return EOF;
  if (_IO_in_backup (fp))
    fp->_IO_read_base = fp->_IO_backup_base;
  else
    {
      fp->_IO_read_base = fp->_IO_buf_base;
      if (fp->_IO_write_ptr > fp->_IO_read_end)
	fp->_IO_read_end = fp->_IO_write_ptr;
    }
  fp->_IO_read_ptr = fp->_IO_write_ptr;

  fp->_IO_write_base = fp->_IO_write_ptr = fp->_IO_write_end = fp->_IO_read_ptr;

  fp->_flags &= ~_IO_CURRENTLY_PUTTING;
  return 0;
}
libc_hidden_def (_IO_switch_to_get_mode)

void
_IO_free_backup_area (FILE *fp)
{
  if (_IO_in_backup (fp))
    _IO_switch_to_main_get_area (fp);  /* Just in case. */
  free (fp->_IO_save_base);
  fp->_IO_save_base = NULL;
  fp->_IO_save_end = NULL;
  fp->_IO_backup_base = NULL;
}
libc_hidden_def (_IO_free_backup_area)

int
__overflow (FILE *f, int ch)
{
  /* This is a single-byte stream.  */
  if (f->_mode == 0)
    _IO_fwide (f, -1);
  return _IO_OVERFLOW (f, ch);
}
libc_hidden_def (__overflow)

static int
save_for_backup (FILE *fp, char *end_p)
{
  /* Append [_IO_read_base..end_p] to backup area. */
  ssize_t least_mark = _IO_least_marker (fp, end_p);
  /* needed_size is how much space we need in the backup area. */
  size_t needed_size = (end_p - fp->_IO_read_base) - least_mark;
  /* FIXME: Dubious arithmetic if pointers are NULL */
  size_t current_Bsize = fp->_IO_save_end - fp->_IO_save_base;
  size_t avail; /* Extra space available for future expansion. */
  ssize_t delta;
  struct _IO_marker *mark;
  if (needed_size > current_Bsize)
    {
      char *new_buffer;
      avail = 100;
      new_buffer = (char *) malloc (avail + needed_size);
      if (new_buffer == NULL)
	return EOF;		/* FIXME */
      if (least_mark < 0)
	{
	  __mempcpy (__mempcpy (new_buffer + avail,
				fp->_IO_save_end + least_mark,
				-least_mark),
		     fp->_IO_read_base,
		     end_p - fp->_IO_read_base);
	}
      else
	memcpy (new_buffer + avail,
		fp->_IO_read_base + least_mark,
		needed_size);
      free (fp->_IO_save_base);
      fp->_IO_save_base = new_buffer;
      fp->_IO_save_end = new_buffer + avail + needed_size;
    }
  else
    {
      avail = current_Bsize - needed_size;
      if (least_mark < 0)
	{
	  memmove (fp->_IO_save_base + avail,
		   fp->_IO_save_end + least_mark,
		   -least_mark);
	  memcpy (fp->_IO_save_base + avail - least_mark,
		  fp->_IO_read_base,
		  end_p - fp->_IO_read_base);
	}
      else if (needed_size > 0)
	memcpy (fp->_IO_save_base + avail,
		fp->_IO_read_base + least_mark,
		needed_size);
    }
  fp->_IO_backup_base = fp->_IO_save_base + avail;
  /* Adjust all the streammarkers. */
  delta = end_p - fp->_IO_read_base;
  for (mark = fp->_markers; mark != NULL; mark = mark->_next)
    mark->_pos -= delta;
  return 0;
}

int
__underflow (FILE *fp)
{
  if (_IO_vtable_offset (fp) == 0 && _IO_fwide (fp, -1) != -1)
    return EOF;

  if (fp->_mode == 0)
    _IO_fwide (fp, -1);
  if (_IO_in_put_mode (fp))
    if (_IO_switch_to_get_mode (fp) == EOF)
      return EOF;
  if (fp->_IO_read_ptr < fp->_IO_read_end)
    return *(unsigned char *) fp->_IO_read_ptr;
  if (_IO_in_backup (fp))
    {
      _IO_switch_to_main_get_area (fp);
      if (fp->_IO_read_ptr < fp->_IO_read_end)
	return *(unsigned char *) fp->_IO_read_ptr;
    }
  if (_IO_have_markers (fp))
    {
      if (save_for_backup (fp, fp->_IO_read_end))
	return EOF;
    }
  else if (_IO_have_backup (fp))
    _IO_free_backup_area (fp);
  return _IO_UNDERFLOW (fp);
}
libc_hidden_def (__underflow)

int
__uflow (FILE *fp)
{
  if (_IO_vtable_offset (fp) == 0 && _IO_fwide (fp, -1) != -1)
    return EOF;

  if (fp->_mode == 0)
    _IO_fwide (fp, -1);
  if (_IO_in_put_mode (fp))
    if (_IO_switch_to_get_mode (fp) == EOF)
      return EOF;
  if (fp->_IO_read_ptr < fp->_IO_read_end)
    return *(unsigned char *) fp->_IO_read_ptr++;
  if (_IO_in_backup (fp))
    {
      _IO_switch_to_main_get_area (fp);
      if (fp->_IO_read_ptr < fp->_IO_read_end)
	return *(unsigned char *) fp->_IO_read_ptr++;
    }
  if (_IO_have_markers (fp))
    {
      if (save_for_backup (fp, fp->_IO_read_end))
	return EOF;
    }
  else if (_IO_have_backup (fp))
    _IO_free_backup_area (fp);
  return _IO_UFLOW (fp);
}
libc_hidden_def (__uflow)

void
_IO_setb (FILE *f, char *b, char *eb, int a)
{
  if (f->_IO_buf_base && !(f->_flags & _IO_USER_BUF))
    free (f->_IO_buf_base);
  f->_IO_buf_base = b;
  f->_IO_buf_end = eb;
  if (a)
    f->_flags &= ~_IO_USER_BUF;
  else
    f->_flags |= _IO_USER_BUF;
}
libc_hidden_def (_IO_setb)

void
_IO_doallocbuf (FILE *fp)
{
  if (fp->_IO_buf_base)
    return;
  if (!(fp->_flags & _IO_UNBUFFERED) || fp->_mode > 0)
    if (_IO_DOALLOCATE (fp) != EOF)
      return;
  _IO_setb (fp, fp->_shortbuf, fp->_shortbuf+1, 0);
}
libc_hidden_def (_IO_doallocbuf)

int
_IO_default_underflow (FILE *fp)
{
  return EOF;
}

int
_IO_default_uflow (FILE *fp)
{
  int ch = _IO_UNDERFLOW (fp);
  if (ch == EOF)
    return EOF;
  return *(unsigned char *) fp->_IO_read_ptr++;
}
libc_hidden_def (_IO_default_uflow)

size_t
_IO_default_xsputn (FILE *f, const void *data, size_t n)
{
  const char *s = (char *) data;
  size_t more = n;
  if (more <= 0)
    return 0;
  for (;;)
    {
      /* Space available. */
      if (f->_IO_write_ptr < f->_IO_write_end)
	{
	  size_t count = f->_IO_write_end - f->_IO_write_ptr;
	  if (count > more)
	    count = more;
	  if (count > 20)
	    {
	      f->_IO_write_ptr = __mempcpy (f->_IO_write_ptr, s, count);
	      s += count;
	    }
	  else if (count)
	    {
	      char *p = f->_IO_write_ptr;
	      ssize_t i;
	      for (i = count; --i >= 0; )
		*p++ = *s++;
	      f->_IO_write_ptr = p;
	    }
	  more -= count;
	}
      if (more == 0 || _IO_OVERFLOW (f, (unsigned char) *s++) == EOF)
	break;
      more--;
    }
  return n - more;
}
libc_hidden_def (_IO_default_xsputn)

size_t
_IO_sgetn (FILE *fp, void *data, size_t n)
{
  /* FIXME handle putback buffer here! */
  return _IO_XSGETN (fp, data, n);
}
libc_hidden_def (_IO_sgetn)

size_t
_IO_default_xsgetn (FILE *fp, void *data, size_t n)
{
  size_t more = n;
  char *s = (char*) data;
  for (;;)
    {
      /* Data available. */
      if (fp->_IO_read_ptr < fp->_IO_read_end)
	{
	  size_t count = fp->_IO_read_end - fp->_IO_read_ptr;
	  if (count > more)
	    count = more;
	  if (count > 20)
	    {
	      s = __mempcpy (s, fp->_IO_read_ptr, count);
	      fp->_IO_read_ptr += count;
	    }
	  else if (count)
	    {
	      char *p = fp->_IO_read_ptr;
	      int i = (int) count;
	      while (--i >= 0)
		*s++ = *p++;
	      fp->_IO_read_ptr = p;
	    }
	    more -= count;
	}
      if (more == 0 || __underflow (fp) == EOF)
	break;
    }
  return n - more;
}
libc_hidden_def (_IO_default_xsgetn)

FILE *
_IO_default_setbuf (FILE *fp, char *p, ssize_t len)
{
    if (_IO_SYNC (fp) == EOF)
	return NULL;
    if (p == NULL || len == 0)
      {
	fp->_flags |= _IO_UNBUFFERED;
	_IO_setb (fp, fp->_shortbuf, fp->_shortbuf+1, 0);
      }
    else
      {
	fp->_flags &= ~_IO_UNBUFFERED;
	_IO_setb (fp, p, p+len, 0);
      }
    fp->_IO_write_base = fp->_IO_write_ptr = fp->_IO_write_end = 0;
    fp->_IO_read_base = fp->_IO_read_ptr = fp->_IO_read_end = 0;
    return fp;
}

off64_t
_IO_default_seekpos (FILE *fp, off64_t pos, int mode)
{
  return _IO_SEEKOFF (fp, pos, 0, mode);
}

int
_IO_default_doallocate (FILE *fp)
{
  char *buf;

  buf = malloc(BUFSIZ);
  if (__glibc_unlikely (buf == NULL))
    return EOF;

  _IO_setb (fp, buf, buf+BUFSIZ, 1);
  return 1;
}
libc_hidden_def (_IO_default_doallocate)

void
_IO_init_internal (FILE *fp, int flags)
{
  _IO_no_init (fp, flags, -1, NULL, NULL);
}

void
_IO_init (FILE *fp, int flags)
{
  IO_set_accept_foreign_vtables (&_IO_vtable_check);
  _IO_init_internal (fp, flags);
}

static int stdio_needs_locking;

/* In a single-threaded process most stdio locks can be omitted.  After
   _IO_enable_locks is called, locks are not optimized away any more.
   It must be first called while the process is still single-threaded.

   This lock optimization can be disabled on a per-file basis by setting
   _IO_FLAGS2_NEED_LOCK, because a file can have user-defined callbacks
   or can be locked with flockfile and then a thread may be created
   between a lock and unlock, so omitting the lock is not valid.

   Here we have to make sure that the flag is set on all existing files
   and files created later.  */
void
_IO_enable_locks (void)
{
  _IO_ITER i;

  if (stdio_needs_locking)
    return;
  stdio_needs_locking = 1;
  for (i = _IO_iter_begin (); i != _IO_iter_end (); i = _IO_iter_next (i))
    _IO_iter_file (i)->_flags2 |= _IO_FLAGS2_NEED_LOCK;
}
libc_hidden_def (_IO_enable_locks)

void
_IO_old_init (FILE *fp, int flags)
{
  fp->_flags = _IO_MAGIC|flags;
  fp->_flags2 = 0;
  if (stdio_needs_locking)
    fp->_flags2 |= _IO_FLAGS2_NEED_LOCK;
  fp->_IO_buf_base = NULL;
  fp->_IO_buf_end = NULL;
  fp->_IO_read_base = NULL;
  fp->_IO_read_ptr = NULL;
  fp->_IO_read_end = NULL;
  fp->_IO_write_base = NULL;
  fp->_IO_write_ptr = NULL;
  fp->_IO_write_end = NULL;
  fp->_chain = NULL; /* Not necessary. */

  fp->_IO_save_base = NULL;
  fp->_IO_backup_base = NULL;
  fp->_IO_save_end = NULL;
  fp->_markers = NULL;
  fp->_cur_column = 0;
#if _IO_JUMPS_OFFSET
  fp->_vtable_offset = 0;
#endif
#ifdef _IO_MTSAFE_IO
  if (fp->_lock != NULL)
    _IO_lock_init (*fp->_lock);
#endif
}

void
_IO_no_init (FILE *fp, int flags, int orientation,
	     struct _IO_wide_data *wd, const struct _IO_jump_t *jmp)
{
  _IO_old_init (fp, flags);
  fp->_mode = orientation;
  if (orientation >= 0)
    {
      fp->_wide_data = wd;
      fp->_wide_data->_IO_buf_base = NULL;
      fp->_wide_data->_IO_buf_end = NULL;
      fp->_wide_data->_IO_read_base = NULL;
      fp->_wide_data->_IO_read_ptr = NULL;
      fp->_wide_data->_IO_read_end = NULL;
      fp->_wide_data->_IO_write_base = NULL;
      fp->_wide_data->_IO_write_ptr = NULL;
      fp->_wide_data->_IO_write_end = NULL;
      fp->_wide_data->_IO_save_base = NULL;
      fp->_wide_data->_IO_backup_base = NULL;
      fp->_wide_data->_IO_save_end = NULL;

      fp->_wide_data->_wide_vtable = jmp;
    }
  else
    /* Cause predictable crash when a wide function is called on a byte
       stream.  */
    fp->_wide_data = (struct _IO_wide_data *) -1L;
  fp->_freeres_list = NULL;
}

int
_IO_default_sync (FILE *fp)
{
  return 0;
}

/* The way the C++ classes are mapped into the C functions in the
   current implementation, this function can get called twice! */

void
_IO_default_finish (FILE *fp, int dummy)
{
  struct _IO_marker *mark;
  if (fp->_IO_buf_base && !(fp->_flags & _IO_USER_BUF))
    {
      free (fp->_IO_buf_base);
      fp->_IO_buf_base = fp->_IO_buf_end = NULL;
    }

  for (mark = fp->_markers; mark != NULL; mark = mark->_next)
    mark->_sbuf = NULL;

  if (fp->_IO_save_base)
    {
      free (fp->_IO_save_base);
      fp->_IO_save_base = NULL;
    }

  _IO_un_link ((struct _IO_FILE_plus *) fp);

#ifdef _IO_MTSAFE_IO
  if (fp->_lock != NULL)
    _IO_lock_fini (*fp->_lock);
#endif
}
libc_hidden_def (_IO_default_finish)

off64_t
_IO_default_seekoff (FILE *fp, off64_t offset, int dir, int mode)
{
  return _IO_pos_BAD;
}

int
_IO_sputbackc (FILE *fp, int c)
{
  int result;

  if (fp->_IO_read_ptr > fp->_IO_read_base
      && (unsigned char)fp->_IO_read_ptr[-1] == (unsigned char)c)
    {
      fp->_IO_read_ptr--;
      result = (unsigned char) c;
    }
  else
    result = _IO_PBACKFAIL (fp, c);

  if (result != EOF)
    fp->_flags &= ~_IO_EOF_SEEN;

  return result;
}
libc_hidden_def (_IO_sputbackc)

int
_IO_sungetc (FILE *fp)
{
  int result;

  if (fp->_IO_read_ptr > fp->_IO_read_base)
    {
      fp->_IO_read_ptr--;
      result = (unsigned char) *fp->_IO_read_ptr;
    }
  else
    result = _IO_PBACKFAIL (fp, EOF);

  if (result != EOF)
    fp->_flags &= ~_IO_EOF_SEEN;

  return result;
}

unsigned
_IO_adjust_column (unsigned start, const char *line, int count)
{
  const char *ptr = line + count;
  while (ptr > line)
    if (*--ptr == '\n')
      return line + count - ptr - 1;
  return start + count;
}
libc_hidden_def (_IO_adjust_column)

int
_IO_flush_all_lockp (int do_lock)
{
  int result = 0;
  FILE *fp;

#ifdef _IO_MTSAFE_IO
  _IO_cleanup_region_start_noarg (flush_cleanup);
  _IO_lock_lock (list_all_lock);
#endif

  for (fp = (FILE *) _IO_list_all; fp != NULL; fp = fp->_chain)
    {
      run_fp = fp;
      if (do_lock)
	_IO_flockfile (fp);

      if (((fp->_mode <= 0 && fp->_IO_write_ptr > fp->_IO_write_base)
	   || (_IO_vtable_offset (fp) == 0
	       && fp->_mode > 0 && (fp->_wide_data->_IO_write_ptr
				    > fp->_wide_data->_IO_write_base))
	   )
	  && _IO_OVERFLOW (fp, EOF) == EOF)
	result = EOF;

      if (do_lock)
	_IO_funlockfile (fp);
      run_fp = NULL;
    }

#ifdef _IO_MTSAFE_IO
  _IO_lock_unlock (list_all_lock);
  _IO_cleanup_region_end (0);
#endif

  return result;
}


int
_IO_flush_all (void)
{
  /* We want locking.  */
  return _IO_flush_all_lockp (1);
}
libc_hidden_def (_IO_flush_all)

void
_IO_flush_all_linebuffered (void)
{
  FILE *fp;

#ifdef _IO_MTSAFE_IO
  _IO_cleanup_region_start_noarg (flush_cleanup);
  _IO_lock_lock (list_all_lock);
#endif

  for (fp = (FILE *) _IO_list_all; fp != NULL; fp = fp->_chain)
    {
      run_fp = fp;
      _IO_flockfile (fp);

      if ((fp->_flags & _IO_NO_WRITES) == 0 && fp->_flags & _IO_LINE_BUF)
	_IO_OVERFLOW (fp, EOF);

      _IO_funlockfile (fp);
      run_fp = NULL;
    }

#ifdef _IO_MTSAFE_IO
  _IO_lock_unlock (list_all_lock);
  _IO_cleanup_region_end (0);
#endif
}
libc_hidden_def (_IO_flush_all_linebuffered)
weak_alias (_IO_flush_all_linebuffered, _flushlbf)


/* The following is a bit tricky.  In general, we want to unbuffer the
   streams so that all output which follows is seen.  If we are not
   looking for memory leaks it does not make much sense to free the
   actual buffer because this will happen anyway once the program
   terminated.  If we do want to look for memory leaks we have to free
   the buffers.  Whether something is freed is determined by the
   function sin the libc_freeres section.  Those are called as part of
   the atexit routine, just like _IO_cleanup.  The problem is we do
   not know whether the freeres code is called first or _IO_cleanup.
   if the former is the case, we set the DEALLOC_BUFFER variable to
   true and _IO_unbuffer_all will take care of the rest.  If
   _IO_unbuffer_all is called first we add the streams to a list
   which the freeres function later can walk through.  */
static void _IO_unbuffer_all (void);

static bool dealloc_buffers;
static FILE *freeres_list;

static void
_IO_unbuffer_all (void)
{
  FILE *fp;

#ifdef _IO_MTSAFE_IO
  _IO_cleanup_region_start_noarg (flush_cleanup);
  _IO_lock_lock (list_all_lock);
#endif

  for (fp = (FILE *) _IO_list_all; fp; fp = fp->_chain)
    {
      int legacy = 0;

#if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_1)
      if (__glibc_unlikely (_IO_vtable_offset (fp) != 0))
	legacy = 1;
#endif

      if (! (fp->_flags & _IO_UNBUFFERED)
	  /* Iff stream is un-orientated, it wasn't used. */
	  && (legacy || fp->_mode != 0))
	{
#ifdef _IO_MTSAFE_IO
	  int cnt;
#define MAXTRIES 2
	  for (cnt = 0; cnt < MAXTRIES; ++cnt)
	    if (fp->_lock == NULL || _IO_lock_trylock (*fp->_lock) == 0)
	      break;
	    else
	      /* Give the other thread time to finish up its use of the
		 stream.  */
	      __sched_yield ();
#endif

	  if (! legacy && ! dealloc_buffers && !(fp->_flags & _IO_USER_BUF))
	    {
	      fp->_flags |= _IO_USER_BUF;

	      fp->_freeres_list = freeres_list;
	      freeres_list = fp;
	      fp->_freeres_buf = fp->_IO_buf_base;
	    }

	  _IO_SETBUF (fp, NULL, 0);

	  if (! legacy && fp->_mode > 0)
	    _IO_wsetb (fp, NULL, NULL, 0);

#ifdef _IO_MTSAFE_IO
	  if (cnt < MAXTRIES && fp->_lock != NULL)
	    _IO_lock_unlock (*fp->_lock);
#endif
	}

      /* Make sure that never again the wide char functions can be
	 used.  */
      if (! legacy)
	fp->_mode = -1;
    }

#ifdef _IO_MTSAFE_IO
  _IO_lock_unlock (list_all_lock);
  _IO_cleanup_region_end (0);
#endif
}


libc_freeres_fn (buffer_free)
{
  dealloc_buffers = true;

  while (freeres_list != NULL)
    {
      free (freeres_list->_freeres_buf);

      freeres_list = freeres_list->_freeres_list;
    }
}


int
_IO_cleanup (void)
{
  /* We do *not* want locking.  Some threads might use streams but
     that is their problem, we flush them underneath them.  */
  int result = _IO_flush_all_lockp (0);

  /* We currently don't have a reliable mechanism for making sure that
     C++ static destructors are executed in the correct order.
     So it is possible that other static destructors might want to
     write to cout - and they're supposed to be able to do so.

     The following will make the standard streambufs be unbuffered,
     which forces any output from late destructors to be written out. */
  _IO_unbuffer_all ();

  return result;
}


void
_IO_init_marker (struct _IO_marker *marker, FILE *fp)
{
  marker->_sbuf = fp;
  if (_IO_in_put_mode (fp))
    _IO_switch_to_get_mode (fp);
  if (_IO_in_backup (fp))
    marker->_pos = fp->_IO_read_ptr - fp->_IO_read_end;
  else
    marker->_pos = fp->_IO_read_ptr - fp->_IO_read_base;

  /* Should perhaps sort the chain? */
  marker->_next = fp->_markers;
  fp->_markers = marker;
}

void
_IO_remove_marker (struct _IO_marker *marker)
{
  /* Unlink from sb's chain. */
  struct _IO_marker **ptr = &marker->_sbuf->_markers;
  for (; ; ptr = &(*ptr)->_next)
    {
      if (*ptr == NULL)
	break;
      else if (*ptr == marker)
	{
	  *ptr = marker->_next;
	  return;
	}
    }
  /* FIXME: if _sbuf has a backup area that is no longer needed,
     should we delete it now, or wait until the next underflow? */
}

#define BAD_DELTA EOF

int
_IO_marker_difference (struct _IO_marker *mark1, struct _IO_marker *mark2)
{
  return mark1->_pos - mark2->_pos;
}

/* Return difference between MARK and current position of MARK's stream. */
int
_IO_marker_delta (struct _IO_marker *mark)
{
  int cur_pos;
  if (mark->_sbuf == NULL)
    return BAD_DELTA;
  if (_IO_in_backup (mark->_sbuf))
    cur_pos = mark->_sbuf->_IO_read_ptr - mark->_sbuf->_IO_read_end;
  else
    cur_pos = mark->_sbuf->_IO_read_ptr - mark->_sbuf->_IO_read_base;
  return mark->_pos - cur_pos;
}

int
_IO_seekmark (FILE *fp, struct _IO_marker *mark, int delta)
{
  if (mark->_sbuf != fp)
    return EOF;
 if (mark->_pos >= 0)
    {
      if (_IO_in_backup (fp))
	_IO_switch_to_main_get_area (fp);
      fp->_IO_read_ptr = fp->_IO_read_base + mark->_pos;
    }
  else
    {
      if (!_IO_in_backup (fp))
	_IO_switch_to_backup_area (fp);
      fp->_IO_read_ptr = fp->_IO_read_end + mark->_pos;
    }
  return 0;
}

void
_IO_unsave_markers (FILE *fp)
{
  struct _IO_marker *mark = fp->_markers;
  if (mark)
    {
      fp->_markers = 0;
    }

  if (_IO_have_backup (fp))
    _IO_free_backup_area (fp);
}
libc_hidden_def (_IO_unsave_markers)

int
_IO_default_pbackfail (FILE *fp, int c)
{
  if (fp->_IO_read_ptr > fp->_IO_read_base && !_IO_in_backup (fp)
      && (unsigned char) fp->_IO_read_ptr[-1] == c)
    --fp->_IO_read_ptr;
  else
    {
      /* Need to handle a filebuf in write mode (switch to read mode). FIXME!*/
      if (!_IO_in_backup (fp))
	{
	  /* We need to keep the invariant that the main get area
	     logically follows the backup area.  */
	  if (fp->_IO_read_ptr > fp->_IO_read_base && _IO_have_backup (fp))
	    {
	      if (save_for_backup (fp, fp->_IO_read_ptr))
		return EOF;
	    }
	  else if (!_IO_have_backup (fp))
	    {
	      /* No backup buffer: allocate one. */
	      /* Use nshort buffer, if unused? (probably not)  FIXME */
	      int backup_size = 128;
	      char *bbuf = (char *) malloc (backup_size);
	      if (bbuf == NULL)
		return EOF;
	      fp->_IO_save_base = bbuf;
	      fp->_IO_save_end = fp->_IO_save_base + backup_size;
	      fp->_IO_backup_base = fp->_IO_save_end;
	    }
	  fp->_IO_read_base = fp->_IO_read_ptr;
	  _IO_switch_to_backup_area (fp);
	}
      else if (fp->_IO_read_ptr <= fp->_IO_read_base)
	{
	  /* Increase size of existing backup buffer. */
	  size_t new_size;
	  size_t old_size = fp->_IO_read_end - fp->_IO_read_base;
	  char *new_buf;
	  new_size = 2 * old_size;
	  new_buf = (char *) malloc (new_size);
	  if (new_buf == NULL)
	    return EOF;
	  memcpy (new_buf + (new_size - old_size), fp->_IO_read_base,
		  old_size);
	  free (fp->_IO_read_base);
	  _IO_setg (fp, new_buf, new_buf + (new_size - old_size),
		    new_buf + new_size);
	  fp->_IO_backup_base = fp->_IO_read_ptr;
	}

      *--fp->_IO_read_ptr = c;
    }
  return (unsigned char) c;
}
libc_hidden_def (_IO_default_pbackfail)

off64_t
_IO_default_seek (FILE *fp, off64_t offset, int dir)
{
  return _IO_pos_BAD;
}

int
_IO_default_stat (FILE *fp, void *st)
{
  return EOF;
}

ssize_t
_IO_default_read (FILE *fp, void *data, ssize_t n)
{
  return -1;
}

ssize_t
_IO_default_write (FILE *fp, const void *data, ssize_t n)
{
  return 0;
}

int
_IO_default_showmanyc (FILE *fp)
{
  return -1;
}

void
_IO_default_imbue (FILE *fp, void *locale)
{
}

_IO_ITER
_IO_iter_begin (void)
{
  return (_IO_ITER) _IO_list_all;
}
libc_hidden_def (_IO_iter_begin)

_IO_ITER
_IO_iter_end (void)
{
  return NULL;
}
libc_hidden_def (_IO_iter_end)

_IO_ITER
_IO_iter_next (_IO_ITER iter)
{
  return iter->_chain;
}
libc_hidden_def (_IO_iter_next)

FILE *
_IO_iter_file (_IO_ITER iter)
{
  return iter;
}
libc_hidden_def (_IO_iter_file)

void
_IO_list_lock (void)
{
#ifdef _IO_MTSAFE_IO
  _IO_lock_lock (list_all_lock);
#endif
}
libc_hidden_def (_IO_list_lock)

void
_IO_list_unlock (void)
{
#ifdef _IO_MTSAFE_IO
  _IO_lock_unlock (list_all_lock);
#endif
}
libc_hidden_def (_IO_list_unlock)

void
_IO_list_resetlock (void)
{
#ifdef _IO_MTSAFE_IO
  _IO_lock_init (list_all_lock);
#endif
}
libc_hidden_def (_IO_list_resetlock)

text_set_element(__libc_atexit, _IO_cleanup);
