/* Copyright (C) 1995-2021 Free Software Foundation, Inc.
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

#include "libioP.h"
#include "strfile.h"
#include <stdio.h>
#include <stdlib.h>


struct _IO_FILE_memstream
{
  _IO_strfile _sf;
  char **bufloc;
  size_t *sizeloc;
};


static int _IO_mem_sync (FILE* fp) __THROW;
static void _IO_mem_finish (FILE* fp, int) __THROW;


static const struct _IO_jump_t _IO_mem_jumps libio_vtable =
{
  JUMP_INIT_DUMMY,
  JUMP_INIT (finish, _IO_mem_finish),
  JUMP_INIT (overflow, _IO_str_overflow),
  JUMP_INIT (underflow, _IO_str_underflow),
  JUMP_INIT (uflow, _IO_default_uflow),
  JUMP_INIT (pbackfail, _IO_str_pbackfail),
  JUMP_INIT (xsputn, _IO_default_xsputn),
  JUMP_INIT (xsgetn, _IO_default_xsgetn),
  JUMP_INIT (seekoff, _IO_str_seekoff),
  JUMP_INIT (seekpos, _IO_default_seekpos),
  JUMP_INIT (setbuf, _IO_default_setbuf),
  JUMP_INIT (sync, _IO_mem_sync),
  JUMP_INIT (doallocate, _IO_default_doallocate),
  JUMP_INIT (read, _IO_default_read),
  JUMP_INIT (write, _IO_default_write),
  JUMP_INIT (seek, _IO_default_seek),
  JUMP_INIT (close, _IO_default_close),
  JUMP_INIT (stat, _IO_default_stat),
  JUMP_INIT(showmanyc, _IO_default_showmanyc),
  JUMP_INIT(imbue, _IO_default_imbue)
};

/* Open a stream that writes into a malloc'd buffer that is expanded as
   necessary.  *BUFLOC and *SIZELOC are updated with the buffer's location
   and the number of characters written on fflush or fclose.  */
FILE *
__open_memstream (char **bufloc, size_t *sizeloc)
{
  struct locked_FILE
  {
    struct _IO_FILE_memstream fp;
#ifdef _IO_MTSAFE_IO
    _IO_lock_t lock;
#endif
    struct _IO_wide_data wd;
  } *new_f;
  char *buf;

  new_f = (struct locked_FILE *) malloc (sizeof (struct locked_FILE));
  if (new_f == NULL)
    return NULL;
#ifdef _IO_MTSAFE_IO
  new_f->fp._sf._sbf._f._lock = &new_f->lock;
#endif

  buf = calloc (1, BUFSIZ);
  if (buf == NULL)
    {
      free (new_f);
      return NULL;
    }
  _IO_init_internal (&new_f->fp._sf._sbf._f, 0);
  _IO_JUMPS_FILE_plus (&new_f->fp._sf._sbf) = &_IO_mem_jumps;
  _IO_str_init_static_internal (&new_f->fp._sf, buf, BUFSIZ, buf);
  new_f->fp._sf._sbf._f._flags &= ~_IO_USER_BUF;
  new_f->fp._sf._s._allocate_buffer_unused = (_IO_alloc_type) malloc;
  new_f->fp._sf._s._free_buffer_unused = (_IO_free_type) free;

  new_f->fp.bufloc = bufloc;
  new_f->fp.sizeloc = sizeloc;

  /* Disable single thread optimization.  BZ 21735.  */
  new_f->fp._sf._sbf._f._flags2 |= _IO_FLAGS2_NEED_LOCK;

  return (FILE *) &new_f->fp._sf._sbf;
}
libc_hidden_def (__open_memstream)
weak_alias (__open_memstream, open_memstream)


static int
_IO_mem_sync (FILE *fp)
{
  struct _IO_FILE_memstream *mp = (struct _IO_FILE_memstream *) fp;

  if (fp->_IO_write_ptr == fp->_IO_write_end)
    {
      _IO_str_overflow (fp, '\0');
      --fp->_IO_write_ptr;
    }

  *mp->bufloc = fp->_IO_write_base;
  *mp->sizeloc = fp->_IO_write_ptr - fp->_IO_write_base;

  return 0;
}


static void
_IO_mem_finish (FILE *fp, int dummy)
{
  struct _IO_FILE_memstream *mp = (struct _IO_FILE_memstream *) fp;

  *mp->bufloc = (char *) realloc (fp->_IO_write_base,
				  fp->_IO_write_ptr - fp->_IO_write_base + 1);
  if (*mp->bufloc != NULL)
    {
      (*mp->bufloc)[fp->_IO_write_ptr - fp->_IO_write_base] = '\0';
      *mp->sizeloc = fp->_IO_write_ptr - fp->_IO_write_base;

      fp->_IO_buf_base = NULL;
    }

  _IO_str_finish (fp, 0);
}
