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

#include <assert.h>
#include "strfile.h"
#include "libioP.h"
#include <string.h>
#include <wchar.h>
#include <stdio_ext.h>

void
_IO_wstr_init_static (FILE *fp, wchar_t *ptr, size_t size,
		      wchar_t *pstart)
{
  wchar_t *end;

  if (size == 0)
    end = ptr + __wcslen (ptr);
  else if ((size_t) ptr + size * sizeof (wchar_t) > (size_t) ptr)
    end = ptr + size;
  else
    /* Even for misaligned ptr make sure there is integral number of wide
       characters.  */
    end = ptr + (-1 - (size_t) ptr) / sizeof (wchar_t);
  _IO_wsetb (fp, ptr, end, 0);

  fp->_wide_data->_IO_write_base = ptr;
  fp->_wide_data->_IO_read_base = ptr;
  fp->_wide_data->_IO_read_ptr = ptr;
  if (pstart)
    {
      fp->_wide_data->_IO_write_ptr = pstart;
      fp->_wide_data->_IO_write_end = end;
      fp->_wide_data->_IO_read_end = pstart;
    }
  else
    {
      fp->_wide_data->_IO_write_ptr = ptr;
      fp->_wide_data->_IO_write_end = ptr;
      fp->_wide_data->_IO_read_end = end;
    }
  /* A null _allocate_buffer function flags the strfile as being static. */
  (((_IO_strfile *) fp)->_s._allocate_buffer_unused) = (_IO_alloc_type)0;
}

wint_t
_IO_wstr_overflow (FILE *fp, wint_t c)
{
  int flush_only = c == WEOF;
  size_t pos;
  if (fp->_flags & _IO_NO_WRITES)
      return flush_only ? 0 : WEOF;
  if ((fp->_flags & _IO_TIED_PUT_GET) && !(fp->_flags & _IO_CURRENTLY_PUTTING))
    {
      fp->_flags |= _IO_CURRENTLY_PUTTING;
      fp->_wide_data->_IO_write_ptr = fp->_wide_data->_IO_read_ptr;
      fp->_wide_data->_IO_read_ptr = fp->_wide_data->_IO_read_end;
    }
  pos = fp->_wide_data->_IO_write_ptr - fp->_wide_data->_IO_write_base;
  if (pos >= (size_t) (_IO_wblen (fp) + flush_only))
    {
      if (fp->_flags2 & _IO_FLAGS2_USER_WBUF) /* not allowed to enlarge */
	return WEOF;
      else
	{
	  wchar_t *new_buf;
	  wchar_t *old_buf = fp->_wide_data->_IO_buf_base;
	  size_t old_wblen = _IO_wblen (fp);
	  size_t new_size = 2 * old_wblen + 100;

	  if (__glibc_unlikely (new_size < old_wblen)
	      || __glibc_unlikely (new_size > SIZE_MAX / sizeof (wchar_t)))
	    return EOF;

	  new_buf = malloc (new_size * sizeof (wchar_t));
	  if (new_buf == NULL)
	    {
	      /*	  __ferror(fp) = 1; */
	      return WEOF;
	    }
	  if (old_buf)
	    {
	      __wmemcpy (new_buf, old_buf, old_wblen);
	      free (old_buf);
	      /* Make sure _IO_setb won't try to delete _IO_buf_base. */
	      fp->_wide_data->_IO_buf_base = NULL;
	    }

	  __wmemset (new_buf + old_wblen, L'\0', new_size - old_wblen);

	  _IO_wsetb (fp, new_buf, new_buf + new_size, 1);
	  fp->_wide_data->_IO_read_base =
	    new_buf + (fp->_wide_data->_IO_read_base - old_buf);
	  fp->_wide_data->_IO_read_ptr =
	    new_buf + (fp->_wide_data->_IO_read_ptr - old_buf);
	  fp->_wide_data->_IO_read_end =
	    new_buf + (fp->_wide_data->_IO_read_end - old_buf);
	  fp->_wide_data->_IO_write_ptr =
	    new_buf + (fp->_wide_data->_IO_write_ptr - old_buf);

	  fp->_wide_data->_IO_write_base = new_buf;
	  fp->_wide_data->_IO_write_end = fp->_wide_data->_IO_buf_end;
	}
    }

  if (!flush_only)
    *fp->_wide_data->_IO_write_ptr++ = c;
  if (fp->_wide_data->_IO_write_ptr > fp->_wide_data->_IO_read_end)
    fp->_wide_data->_IO_read_end = fp->_wide_data->_IO_write_ptr;
  return c;
}


wint_t
_IO_wstr_underflow (FILE *fp)
{
  if (fp->_wide_data->_IO_write_ptr > fp->_wide_data->_IO_read_end)
    fp->_wide_data->_IO_read_end = fp->_wide_data->_IO_write_ptr;
  if ((fp->_flags & _IO_TIED_PUT_GET) && (fp->_flags & _IO_CURRENTLY_PUTTING))
    {
      fp->_flags &= ~_IO_CURRENTLY_PUTTING;
      fp->_wide_data->_IO_read_ptr = fp->_wide_data->_IO_write_ptr;
      fp->_wide_data->_IO_write_ptr = fp->_wide_data->_IO_write_end;
    }
  if (fp->_wide_data->_IO_read_ptr < fp->_wide_data->_IO_read_end)
    return *fp->_wide_data->_IO_read_ptr;
  else
    return WEOF;
}


/* The size of the valid part of the buffer.  */
ssize_t
_IO_wstr_count (FILE *fp)
{
  struct _IO_wide_data *wd = fp->_wide_data;

  return ((wd->_IO_write_ptr > wd->_IO_read_end
	   ? wd->_IO_write_ptr : wd->_IO_read_end)
	  - wd->_IO_read_base);
}


static int
enlarge_userbuf (FILE *fp, off64_t offset, int reading)
{
  if ((ssize_t) offset <= _IO_wblen (fp))
    return 0;

  struct _IO_wide_data *wd = fp->_wide_data;

  ssize_t oldend = wd->_IO_write_end - wd->_IO_write_base;

  /* Try to enlarge the buffer.  */
  if (fp->_flags2 & _IO_FLAGS2_USER_WBUF)
    /* User-provided buffer.  */
    return 1;

  size_t newsize = offset + 100;
  if (__glibc_unlikely (newsize > SIZE_MAX / sizeof (wchar_t)))
    return 1;

  wchar_t *oldbuf = wd->_IO_buf_base;
  wchar_t *newbuf = malloc (newsize * sizeof (wchar_t));
  if (newbuf == NULL)
    return 1;

  if (oldbuf != NULL)
    {
      __wmemcpy (newbuf, oldbuf, _IO_wblen (fp));
      free (oldbuf);
      /* Make sure _IO_setb won't try to delete
	 _IO_buf_base. */
      wd->_IO_buf_base = NULL;
    }

  _IO_wsetb (fp, newbuf, newbuf + newsize, 1);

  if (reading)
    {
      wd->_IO_write_base = newbuf + (wd->_IO_write_base - oldbuf);
      wd->_IO_write_ptr = newbuf + (wd->_IO_write_ptr - oldbuf);
      wd->_IO_write_end = newbuf + (wd->_IO_write_end - oldbuf);
      wd->_IO_read_ptr = newbuf + (wd->_IO_read_ptr - oldbuf);

      wd->_IO_read_base = newbuf;
      wd->_IO_read_end = wd->_IO_buf_end;
    }
  else
    {
      wd->_IO_read_base = newbuf + (wd->_IO_read_base - oldbuf);
      wd->_IO_read_ptr = newbuf + (wd->_IO_read_ptr - oldbuf);
      wd->_IO_read_end = newbuf + (wd->_IO_read_end - oldbuf);
      wd->_IO_write_ptr = newbuf + (wd->_IO_write_ptr - oldbuf);

      wd->_IO_write_base = newbuf;
      wd->_IO_write_end = wd->_IO_buf_end;
    }

  /* Clear the area between the last write position and th
     new position.  */
  assert (offset >= oldend);
  if (reading)
    __wmemset (wd->_IO_read_base + oldend, L'\0', offset - oldend);
  else
    __wmemset (wd->_IO_write_base + oldend, L'\0', offset - oldend);

  return 0;
}

static void
_IO_wstr_switch_to_get_mode (FILE *fp)
{
  if (_IO_in_backup (fp))
    fp->_wide_data->_IO_read_base = fp->_wide_data->_IO_backup_base;
  else
    {
      fp->_wide_data->_IO_read_base = fp->_wide_data->_IO_buf_base;
      if (fp->_wide_data->_IO_write_ptr > fp->_wide_data->_IO_read_end)
        fp->_wide_data->_IO_read_end = fp->_wide_data->_IO_write_ptr;
    }
  fp->_wide_data->_IO_read_ptr = fp->_wide_data->_IO_write_ptr;
  fp->_wide_data->_IO_read_end = fp->_wide_data->_IO_write_ptr;

  fp->_flags &= ~_IO_CURRENTLY_PUTTING;
}

off64_t
_IO_wstr_seekoff (FILE *fp, off64_t offset, int dir, int mode)
{
  off64_t new_pos;

  if (mode == 0 && (fp->_flags & _IO_TIED_PUT_GET))
    mode = (fp->_flags & _IO_CURRENTLY_PUTTING ? _IOS_OUTPUT : _IOS_INPUT);

  bool was_writing = ((fp->_wide_data->_IO_write_ptr
		       > fp->_wide_data->_IO_write_base)
		     || _IO_in_put_mode (fp));
  if (was_writing)
    _IO_wstr_switch_to_get_mode (fp);

  if (mode == 0)
    {
      new_pos = (fp->_wide_data->_IO_write_ptr
		 - fp->_wide_data->_IO_write_base);
    }
  else
    {
      ssize_t cur_size = _IO_wstr_count (fp);
      new_pos = EOF;

      /* Move the get pointer, if requested. */
      if (mode & _IOS_INPUT)
	{
	  ssize_t base;
	  switch (dir)
	    {
	    case _IO_seek_set:
	      base = 0;
	      break;
	    case _IO_seek_cur:
	      base = (fp->_wide_data->_IO_read_ptr
		     - fp->_wide_data->_IO_read_base);
	      break;
	    default: /* case _IO_seek_end: */
	      base = cur_size;
	      break;
	    }
	  ssize_t maxval = SSIZE_MAX/sizeof (wchar_t) - base;
	  if (offset < -base || offset > maxval)
	    {
	      __set_errno (EINVAL);
	      return EOF;
	    }
	  base += offset;
	  if (base > cur_size
	      && enlarge_userbuf (fp, base, 1) != 0)
	    return EOF;
	  fp->_wide_data->_IO_read_ptr = (fp->_wide_data->_IO_read_base
					  + base);
	  fp->_wide_data->_IO_read_end = (fp->_wide_data->_IO_read_base
					  + cur_size);
	  new_pos = offset;
	}

      /* Move the put pointer, if requested. */
      if (mode & _IOS_OUTPUT)
	{
	  ssize_t base;
	  switch (dir)
	    {
	    case _IO_seek_set:
	      base = 0;
	      break;
	    case _IO_seek_cur:
	      base = (fp->_wide_data->_IO_write_ptr
		     - fp->_wide_data->_IO_write_base);
	      break;
	    default: /* case _IO_seek_end: */
	      base = cur_size;
	      break;
	    }
	  ssize_t maxval = SSIZE_MAX/sizeof (wchar_t) - base;
	  if (offset < -base || offset > maxval)
	    {
	      __set_errno (EINVAL);
	      return EOF;
	    }
	  base += offset;
	  if (base > cur_size
	      && enlarge_userbuf (fp, base, 0) != 0)
	    return EOF;
	  fp->_wide_data->_IO_write_ptr = (fp->_wide_data->_IO_write_base
					   + base);
	  new_pos = base;
	}
    }
  return new_pos;
}

wint_t
_IO_wstr_pbackfail (FILE *fp, wint_t c)
{
  if ((fp->_flags & _IO_NO_WRITES) && c != WEOF)
    return WEOF;
  return _IO_wdefault_pbackfail (fp, c);
}

void
_IO_wstr_finish (FILE *fp, int dummy)
{
  if (fp->_wide_data->_IO_buf_base && !(fp->_flags2 & _IO_FLAGS2_USER_WBUF))
    free (fp->_wide_data->_IO_buf_base);
  fp->_wide_data->_IO_buf_base = NULL;

  _IO_wdefault_finish (fp, 0);
}

const struct _IO_jump_t _IO_wstr_jumps libio_vtable =
{
  JUMP_INIT_DUMMY,
  JUMP_INIT(finish, _IO_wstr_finish),
  JUMP_INIT(overflow, (_IO_overflow_t) _IO_wstr_overflow),
  JUMP_INIT(underflow, (_IO_underflow_t) _IO_wstr_underflow),
  JUMP_INIT(uflow, (_IO_underflow_t) _IO_wdefault_uflow),
  JUMP_INIT(pbackfail, (_IO_pbackfail_t) _IO_wstr_pbackfail),
  JUMP_INIT(xsputn, _IO_wdefault_xsputn),
  JUMP_INIT(xsgetn, _IO_wdefault_xsgetn),
  JUMP_INIT(seekoff, _IO_wstr_seekoff),
  JUMP_INIT(seekpos, _IO_default_seekpos),
  JUMP_INIT(setbuf, _IO_default_setbuf),
  JUMP_INIT(sync, _IO_default_sync),
  JUMP_INIT(doallocate, _IO_wdefault_doallocate),
  JUMP_INIT(read, _IO_default_read),
  JUMP_INIT(write, _IO_default_write),
  JUMP_INIT(seek, _IO_default_seek),
  JUMP_INIT(close, _IO_default_close),
  JUMP_INIT(stat, _IO_default_stat),
  JUMP_INIT(showmanyc, _IO_default_showmanyc),
  JUMP_INIT(imbue, _IO_default_imbue)
};
