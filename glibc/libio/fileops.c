/* Copyright (C) 1993-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Written by Per Bothner <bothner@cygnus.com>.

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


#include "libioP.h"
#include <assert.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include "../wcsmbs/wcsmbsload.h"
#include "../iconv/gconv_charset.h"
#include "../iconv/gconv_int.h"
#include <shlib-compat.h>
#include <not-cancel.h>
#include <kernel-features.h>

extern struct __gconv_trans_data __libio_translit attribute_hidden;

/* An fstream can be in at most one of put mode, get mode, or putback mode.
   Putback mode is a variant of get mode.

   In a filebuf, there is only one current position, instead of two
   separate get and put pointers.  In get mode, the current position
   is that of gptr(); in put mode that of pptr().

   The position in the buffer that corresponds to the position
   in external file system is normally _IO_read_end, except in putback
   mode, when it is _IO_save_end and also when the file is in append mode,
   since switching from read to write mode automatically sends the position in
   the external file system to the end of file.
   If the field _fb._offset is >= 0, it gives the offset in
   the file as a whole corresponding to eGptr(). (?)

   PUT MODE:
   If a filebuf is in put mode, then all of _IO_read_ptr, _IO_read_end,
   and _IO_read_base are equal to each other.  These are usually equal
   to _IO_buf_base, though not necessarily if we have switched from
   get mode to put mode.  (The reason is to maintain the invariant
   that _IO_read_end corresponds to the external file position.)
   _IO_write_base is non-NULL and usually equal to _IO_buf_base.
   We also have _IO_write_end == _IO_buf_end, but only in fully buffered mode.
   The un-flushed character are those between _IO_write_base and _IO_write_ptr.

   GET MODE:
   If a filebuf is in get or putback mode, eback() != egptr().
   In get mode, the unread characters are between gptr() and egptr().
   The OS file position corresponds to that of egptr().

   PUTBACK MODE:
   Putback mode is used to remember "excess" characters that have
   been sputbackc'd in a separate putback buffer.
   In putback mode, the get buffer points to the special putback buffer.
   The unread characters are the characters between gptr() and egptr()
   in the putback buffer, as well as the area between save_gptr()
   and save_egptr(), which point into the original reserve buffer.
   (The pointers save_gptr() and save_egptr() are the values
   of gptr() and egptr() at the time putback mode was entered.)
   The OS position corresponds to that of save_egptr().

   LINE BUFFERED OUTPUT:
   During line buffered output, _IO_write_base==base() && epptr()==base().
   However, ptr() may be anywhere between base() and ebuf().
   This forces a call to filebuf::overflow(int C) on every put.
   If there is more space in the buffer, and C is not a '\n',
   then C is inserted, and pptr() incremented.

   UNBUFFERED STREAMS:
   If a filebuf is unbuffered(), the _shortbuf[1] is used as the buffer.
*/

#define CLOSED_FILEBUF_FLAGS \
  (_IO_IS_FILEBUF+_IO_NO_READS+_IO_NO_WRITES+_IO_TIED_PUT_GET)


void
_IO_new_file_init_internal (struct _IO_FILE_plus *fp)
{
  /* POSIX.1 allows another file handle to be used to change the position
     of our file descriptor.  Hence we actually don't know the actual
     position before we do the first fseek (and until a following fflush). */
  fp->file._offset = _IO_pos_BAD;
  fp->file._flags |= CLOSED_FILEBUF_FLAGS;

  _IO_link_in (fp);
  fp->file._fileno = -1;
}

/* External version of _IO_new_file_init_internal which switches off
   vtable validation.  */
void
_IO_new_file_init (struct _IO_FILE_plus *fp)
{
  IO_set_accept_foreign_vtables (&_IO_vtable_check);
  _IO_new_file_init_internal (fp);
}

int
_IO_new_file_close_it (FILE *fp)
{
  int write_status;
  if (!_IO_file_is_open (fp))
    return EOF;

  if ((fp->_flags & _IO_NO_WRITES) == 0
      && (fp->_flags & _IO_CURRENTLY_PUTTING) != 0)
    write_status = _IO_do_flush (fp);
  else
    write_status = 0;

  _IO_unsave_markers (fp);

  int close_status = ((fp->_flags2 & _IO_FLAGS2_NOCLOSE) == 0
		      ? _IO_SYSCLOSE (fp) : 0);

  /* Free buffer. */
  if (fp->_mode > 0)
    {
      if (_IO_have_wbackup (fp))
	_IO_free_wbackup_area (fp);
      _IO_wsetb (fp, NULL, NULL, 0);
      _IO_wsetg (fp, NULL, NULL, NULL);
      _IO_wsetp (fp, NULL, NULL);
    }
  _IO_setb (fp, NULL, NULL, 0);
  _IO_setg (fp, NULL, NULL, NULL);
  _IO_setp (fp, NULL, NULL);

  _IO_un_link ((struct _IO_FILE_plus *) fp);
  fp->_flags = _IO_MAGIC|CLOSED_FILEBUF_FLAGS;
  fp->_fileno = -1;
  fp->_offset = _IO_pos_BAD;

  return close_status ? close_status : write_status;
}
libc_hidden_ver (_IO_new_file_close_it, _IO_file_close_it)

void
_IO_new_file_finish (FILE *fp, int dummy)
{
  if (_IO_file_is_open (fp))
    {
      _IO_do_flush (fp);
      if (!(fp->_flags & _IO_DELETE_DONT_CLOSE))
	_IO_SYSCLOSE (fp);
    }
  _IO_default_finish (fp, 0);
}
libc_hidden_ver (_IO_new_file_finish, _IO_file_finish)

FILE *
_IO_file_open (FILE *fp, const char *filename, int posix_mode, int prot,
	       int read_write, int is32not64)
{
  int fdesc;
  if (__glibc_unlikely (fp->_flags2 & _IO_FLAGS2_NOTCANCEL))
    fdesc = __open_nocancel (filename,
			     posix_mode | (is32not64 ? 0 : O_LARGEFILE), prot);
  else
    fdesc = __open (filename, posix_mode | (is32not64 ? 0 : O_LARGEFILE), prot);
  if (fdesc < 0)
    return NULL;
  fp->_fileno = fdesc;
  _IO_mask_flags (fp, read_write,_IO_NO_READS+_IO_NO_WRITES+_IO_IS_APPENDING);
  /* For append mode, send the file offset to the end of the file.  Don't
     update the offset cache though, since the file handle is not active.  */
  if ((read_write & (_IO_IS_APPENDING | _IO_NO_READS))
      == (_IO_IS_APPENDING | _IO_NO_READS))
    {
      off64_t new_pos = _IO_SYSSEEK (fp, 0, _IO_seek_end);
      if (new_pos == _IO_pos_BAD && errno != ESPIPE)
	{
	  __close_nocancel (fdesc);
	  return NULL;
	}
    }
  _IO_link_in ((struct _IO_FILE_plus *) fp);
  return fp;
}
libc_hidden_def (_IO_file_open)

FILE *
_IO_new_file_fopen (FILE *fp, const char *filename, const char *mode,
		    int is32not64)
{
  int oflags = 0, omode;
  int read_write;
  int oprot = 0666;
  int i;
  FILE *result;
  const char *cs;
  const char *last_recognized;

  if (_IO_file_is_open (fp))
    return 0;
  switch (*mode)
    {
    case 'r':
      omode = O_RDONLY;
      read_write = _IO_NO_WRITES;
      break;
    case 'w':
      omode = O_WRONLY;
      oflags = O_CREAT|O_TRUNC;
      read_write = _IO_NO_READS;
      break;
    case 'a':
      omode = O_WRONLY;
      oflags = O_CREAT|O_APPEND;
      read_write = _IO_NO_READS|_IO_IS_APPENDING;
      break;
    default:
      __set_errno (EINVAL);
      return NULL;
    }
  last_recognized = mode;
  for (i = 1; i < 7; ++i)
    {
      switch (*++mode)
	{
	case '\0':
	  break;
	case '+':
	  omode = O_RDWR;
	  read_write &= _IO_IS_APPENDING;
	  last_recognized = mode;
	  continue;
	case 'x':
	  oflags |= O_EXCL;
	  last_recognized = mode;
	  continue;
	case 'b':
	  last_recognized = mode;
	  continue;
	case 'm':
	  fp->_flags2 |= _IO_FLAGS2_MMAP;
	  continue;
	case 'c':
	  fp->_flags2 |= _IO_FLAGS2_NOTCANCEL;
	  continue;
	case 'e':
	  oflags |= O_CLOEXEC;
	  fp->_flags2 |= _IO_FLAGS2_CLOEXEC;
	  continue;
	default:
	  /* Ignore.  */
	  continue;
	}
      break;
    }

  result = _IO_file_open (fp, filename, omode|oflags, oprot, read_write,
			  is32not64);

  if (result != NULL)
    {
      /* Test whether the mode string specifies the conversion.  */
      cs = strstr (last_recognized + 1, ",ccs=");
      if (cs != NULL)
	{
	  /* Yep.  Load the appropriate conversions and set the orientation
	     to wide.  */
	  struct gconv_fcts fcts;
	  struct _IO_codecvt *cc;
	  char *endp = __strchrnul (cs + 5, ',');
	  char *ccs = malloc (endp - (cs + 5) + 3);

	  if (ccs == NULL)
	    {
	      int malloc_err = errno;  /* Whatever malloc failed with.  */
	      (void) _IO_file_close_it (fp);
	      __set_errno (malloc_err);
	      return NULL;
	    }

	  *((char *) __mempcpy (ccs, cs + 5, endp - (cs + 5))) = '\0';
	  strip (ccs, ccs);

	  if (__wcsmbs_named_conv (&fcts, ccs[2] == '\0'
				   ? upstr (ccs, cs + 5) : ccs) != 0)
	    {
	      /* Something went wrong, we cannot load the conversion modules.
		 This means we cannot proceed since the user explicitly asked
		 for these.  */
	      (void) _IO_file_close_it (fp);
	      free (ccs);
	      __set_errno (EINVAL);
	      return NULL;
	    }

	  free (ccs);

	  assert (fcts.towc_nsteps == 1);
	  assert (fcts.tomb_nsteps == 1);

	  fp->_wide_data->_IO_read_ptr = fp->_wide_data->_IO_read_end;
	  fp->_wide_data->_IO_write_ptr = fp->_wide_data->_IO_write_base;

	  /* Clear the state.  We start all over again.  */
	  memset (&fp->_wide_data->_IO_state, '\0', sizeof (__mbstate_t));
	  memset (&fp->_wide_data->_IO_last_state, '\0', sizeof (__mbstate_t));

	  cc = fp->_codecvt = &fp->_wide_data->_codecvt;

	  cc->__cd_in.step = fcts.towc;

	  cc->__cd_in.step_data.__invocation_counter = 0;
	  cc->__cd_in.step_data.__internal_use = 1;
	  cc->__cd_in.step_data.__flags = __GCONV_IS_LAST;
	  cc->__cd_in.step_data.__statep = &result->_wide_data->_IO_state;

	  cc->__cd_out.step = fcts.tomb;

	  cc->__cd_out.step_data.__invocation_counter = 0;
	  cc->__cd_out.step_data.__internal_use = 1;
	  cc->__cd_out.step_data.__flags = __GCONV_IS_LAST | __GCONV_TRANSLIT;
	  cc->__cd_out.step_data.__statep = &result->_wide_data->_IO_state;

	  /* From now on use the wide character callback functions.  */
	  _IO_JUMPS_FILE_plus (fp) = fp->_wide_data->_wide_vtable;

	  /* Set the mode now.  */
	  result->_mode = 1;
	}
    }

  return result;
}
libc_hidden_ver (_IO_new_file_fopen, _IO_file_fopen)

FILE *
_IO_new_file_attach (FILE *fp, int fd)
{
  if (_IO_file_is_open (fp))
    return NULL;
  fp->_fileno = fd;
  fp->_flags &= ~(_IO_NO_READS+_IO_NO_WRITES);
  fp->_flags |= _IO_DELETE_DONT_CLOSE;
  /* Get the current position of the file. */
  /* We have to do that since that may be junk. */
  fp->_offset = _IO_pos_BAD;
  int save_errno = errno;
  if (_IO_SEEKOFF (fp, (off64_t)0, _IO_seek_cur, _IOS_INPUT|_IOS_OUTPUT)
      == _IO_pos_BAD && errno != ESPIPE)
    return NULL;
  __set_errno (save_errno);
  return fp;
}
libc_hidden_ver (_IO_new_file_attach, _IO_file_attach)

FILE *
_IO_new_file_setbuf (FILE *fp, char *p, ssize_t len)
{
  if (_IO_default_setbuf (fp, p, len) == NULL)
    return NULL;

  fp->_IO_write_base = fp->_IO_write_ptr = fp->_IO_write_end
    = fp->_IO_buf_base;
  _IO_setg (fp, fp->_IO_buf_base, fp->_IO_buf_base, fp->_IO_buf_base);

  return fp;
}
libc_hidden_ver (_IO_new_file_setbuf, _IO_file_setbuf)


FILE *
_IO_file_setbuf_mmap (FILE *fp, char *p, ssize_t len)
{
  FILE *result;

  /* Change the function table.  */
  _IO_JUMPS_FILE_plus (fp) = &_IO_file_jumps;
  fp->_wide_data->_wide_vtable = &_IO_wfile_jumps;

  /* And perform the normal operation.  */
  result = _IO_new_file_setbuf (fp, p, len);

  /* If the call failed, restore to using mmap.  */
  if (result == NULL)
    {
      _IO_JUMPS_FILE_plus (fp) = &_IO_file_jumps_mmap;
      fp->_wide_data->_wide_vtable = &_IO_wfile_jumps_mmap;
    }

  return result;
}

static size_t new_do_write (FILE *, const char *, size_t);

/* Write TO_DO bytes from DATA to FP.
   Then mark FP as having empty buffers. */

int
_IO_new_do_write (FILE *fp, const char *data, size_t to_do)
{
  return (to_do == 0
	  || (size_t) new_do_write (fp, data, to_do) == to_do) ? 0 : EOF;
}
libc_hidden_ver (_IO_new_do_write, _IO_do_write)

static size_t
new_do_write (FILE *fp, const char *data, size_t to_do)
{
  size_t count;
  if (fp->_flags & _IO_IS_APPENDING)
    /* On a system without a proper O_APPEND implementation,
       you would need to sys_seek(0, SEEK_END) here, but is
       not needed nor desirable for Unix- or Posix-like systems.
       Instead, just indicate that offset (before and after) is
       unpredictable. */
    fp->_offset = _IO_pos_BAD;
  else if (fp->_IO_read_end != fp->_IO_write_base)
    {
      off64_t new_pos
	= _IO_SYSSEEK (fp, fp->_IO_write_base - fp->_IO_read_end, 1);
      if (new_pos == _IO_pos_BAD)
	return 0;
      fp->_offset = new_pos;
    }
  count = _IO_SYSWRITE (fp, data, to_do);
  if (fp->_cur_column && count)
    fp->_cur_column = _IO_adjust_column (fp->_cur_column - 1, data, count) + 1;
  _IO_setg (fp, fp->_IO_buf_base, fp->_IO_buf_base, fp->_IO_buf_base);
  fp->_IO_write_base = fp->_IO_write_ptr = fp->_IO_buf_base;
  fp->_IO_write_end = (fp->_mode <= 0
		       && (fp->_flags & (_IO_LINE_BUF | _IO_UNBUFFERED))
		       ? fp->_IO_buf_base : fp->_IO_buf_end);
  return count;
}

int
_IO_new_file_underflow (FILE *fp)
{
  ssize_t count;

  /* C99 requires EOF to be "sticky".  */
  if (fp->_flags & _IO_EOF_SEEN)
    return EOF;

  if (fp->_flags & _IO_NO_READS)
    {
      fp->_flags |= _IO_ERR_SEEN;
      __set_errno (EBADF);
      return EOF;
    }
  if (fp->_IO_read_ptr < fp->_IO_read_end)
    return *(unsigned char *) fp->_IO_read_ptr;

  if (fp->_IO_buf_base == NULL)
    {
      /* Maybe we already have a push back pointer.  */
      if (fp->_IO_save_base != NULL)
	{
	  free (fp->_IO_save_base);
	  fp->_flags &= ~_IO_IN_BACKUP;
	}
      _IO_doallocbuf (fp);
    }

  /* FIXME This can/should be moved to genops ?? */
  if (fp->_flags & (_IO_LINE_BUF|_IO_UNBUFFERED))
    {
      /* We used to flush all line-buffered stream.  This really isn't
	 required by any standard.  My recollection is that
	 traditional Unix systems did this for stdout.  stderr better
	 not be line buffered.  So we do just that here
	 explicitly.  --drepper */
      _IO_acquire_lock (stdout);

      if ((stdout->_flags & (_IO_LINKED | _IO_NO_WRITES | _IO_LINE_BUF))
	  == (_IO_LINKED | _IO_LINE_BUF))
	_IO_OVERFLOW (stdout, EOF);

      _IO_release_lock (stdout);
    }

  _IO_switch_to_get_mode (fp);

  /* This is very tricky. We have to adjust those
     pointers before we call _IO_SYSREAD () since
     we may longjump () out while waiting for
     input. Those pointers may be screwed up. H.J. */
  fp->_IO_read_base = fp->_IO_read_ptr = fp->_IO_buf_base;
  fp->_IO_read_end = fp->_IO_buf_base;
  fp->_IO_write_base = fp->_IO_write_ptr = fp->_IO_write_end
    = fp->_IO_buf_base;

  count = _IO_SYSREAD (fp, fp->_IO_buf_base,
		       fp->_IO_buf_end - fp->_IO_buf_base);
  if (count <= 0)
    {
      if (count == 0)
	fp->_flags |= _IO_EOF_SEEN;
      else
	fp->_flags |= _IO_ERR_SEEN, count = 0;
  }
  fp->_IO_read_end += count;
  if (count == 0)
    {
      /* If a stream is read to EOF, the calling application may switch active
	 handles.  As a result, our offset cache would no longer be valid, so
	 unset it.  */
      fp->_offset = _IO_pos_BAD;
      return EOF;
    }
  if (fp->_offset != _IO_pos_BAD)
    _IO_pos_adjust (fp->_offset, count);
  return *(unsigned char *) fp->_IO_read_ptr;
}
libc_hidden_ver (_IO_new_file_underflow, _IO_file_underflow)

/* Guts of underflow callback if we mmap the file.  This stats the file and
   updates the stream state to match.  In the normal case we return zero.
   If the file is no longer eligible for mmap, its jump tables are reset to
   the vanilla ones and we return nonzero.  */
static int
mmap_remap_check (FILE *fp)
{
  struct __stat64_t64 st;

  if (_IO_SYSSTAT (fp, &st) == 0
      && S_ISREG (st.st_mode) && st.st_size != 0
      /* Limit the file size to 1MB for 32-bit machines.  */
      && (sizeof (ptrdiff_t) > 4 || st.st_size < 1*1024*1024))
    {
      const size_t pagesize = __getpagesize ();
# define ROUNDED(x)	(((x) + pagesize - 1) & ~(pagesize - 1))
      if (ROUNDED (st.st_size) < ROUNDED (fp->_IO_buf_end
					  - fp->_IO_buf_base))
	{
	  /* We can trim off some pages past the end of the file.  */
	  (void) __munmap (fp->_IO_buf_base + ROUNDED (st.st_size),
			   ROUNDED (fp->_IO_buf_end - fp->_IO_buf_base)
			   - ROUNDED (st.st_size));
	  fp->_IO_buf_end = fp->_IO_buf_base + st.st_size;
	}
      else if (ROUNDED (st.st_size) > ROUNDED (fp->_IO_buf_end
					       - fp->_IO_buf_base))
	{
	  /* The file added some pages.  We need to remap it.  */
	  void *p;
#if _G_HAVE_MREMAP
	  p = __mremap (fp->_IO_buf_base, ROUNDED (fp->_IO_buf_end
						   - fp->_IO_buf_base),
			ROUNDED (st.st_size), MREMAP_MAYMOVE);
	  if (p == MAP_FAILED)
	    {
	      (void) __munmap (fp->_IO_buf_base,
			       fp->_IO_buf_end - fp->_IO_buf_base);
	      goto punt;
	    }
#else
	  (void) __munmap (fp->_IO_buf_base,
			   fp->_IO_buf_end - fp->_IO_buf_base);
	  p = __mmap64 (NULL, st.st_size, PROT_READ, MAP_SHARED,
			fp->_fileno, 0);
	  if (p == MAP_FAILED)
	    goto punt;
#endif
	  fp->_IO_buf_base = p;
	  fp->_IO_buf_end = fp->_IO_buf_base + st.st_size;
	}
      else
	{
	  /* The number of pages didn't change.  */
	  fp->_IO_buf_end = fp->_IO_buf_base + st.st_size;
	}
# undef ROUNDED

      fp->_offset -= fp->_IO_read_end - fp->_IO_read_ptr;
      _IO_setg (fp, fp->_IO_buf_base,
		fp->_offset < fp->_IO_buf_end - fp->_IO_buf_base
		? fp->_IO_buf_base + fp->_offset : fp->_IO_buf_end,
		fp->_IO_buf_end);

      /* If we are already positioned at or past the end of the file, don't
	 change the current offset.  If not, seek past what we have mapped,
	 mimicking the position left by a normal underflow reading into its
	 buffer until EOF.  */

      if (fp->_offset < fp->_IO_buf_end - fp->_IO_buf_base)
	{
	  if (__lseek64 (fp->_fileno, fp->_IO_buf_end - fp->_IO_buf_base,
			 SEEK_SET)
	      != fp->_IO_buf_end - fp->_IO_buf_base)
	    fp->_flags |= _IO_ERR_SEEN;
	  else
	    fp->_offset = fp->_IO_buf_end - fp->_IO_buf_base;
	}

      return 0;
    }
  else
    {
      /* Life is no longer good for mmap.  Punt it.  */
      (void) __munmap (fp->_IO_buf_base,
		       fp->_IO_buf_end - fp->_IO_buf_base);
    punt:
      fp->_IO_buf_base = fp->_IO_buf_end = NULL;
      _IO_setg (fp, NULL, NULL, NULL);
      if (fp->_mode <= 0)
	_IO_JUMPS_FILE_plus (fp) = &_IO_file_jumps;
      else
	_IO_JUMPS_FILE_plus (fp) = &_IO_wfile_jumps;
      fp->_wide_data->_wide_vtable = &_IO_wfile_jumps;

      return 1;
    }
}

/* Special callback replacing the underflow callbacks if we mmap the file.  */
int
_IO_file_underflow_mmap (FILE *fp)
{
  if (fp->_IO_read_ptr < fp->_IO_read_end)
    return *(unsigned char *) fp->_IO_read_ptr;

  if (__glibc_unlikely (mmap_remap_check (fp)))
    /* We punted to the regular file functions.  */
    return _IO_UNDERFLOW (fp);

  if (fp->_IO_read_ptr < fp->_IO_read_end)
    return *(unsigned char *) fp->_IO_read_ptr;

  fp->_flags |= _IO_EOF_SEEN;
  return EOF;
}

static void
decide_maybe_mmap (FILE *fp)
{
  /* We use the file in read-only mode.  This could mean we can
     mmap the file and use it without any copying.  But not all
     file descriptors are for mmap-able objects and on 32-bit
     machines we don't want to map files which are too large since
     this would require too much virtual memory.  */
  struct __stat64_t64 st;

  if (_IO_SYSSTAT (fp, &st) == 0
      && S_ISREG (st.st_mode) && st.st_size != 0
      /* Limit the file size to 1MB for 32-bit machines.  */
      && (sizeof (ptrdiff_t) > 4 || st.st_size < 1*1024*1024)
      /* Sanity check.  */
      && (fp->_offset == _IO_pos_BAD || fp->_offset <= st.st_size))
    {
      /* Try to map the file.  */
      void *p;

      p = __mmap64 (NULL, st.st_size, PROT_READ, MAP_SHARED, fp->_fileno, 0);
      if (p != MAP_FAILED)
	{
	  /* OK, we managed to map the file.  Set the buffer up and use a
	     special jump table with simplified underflow functions which
	     never tries to read anything from the file.  */

	  if (__lseek64 (fp->_fileno, st.st_size, SEEK_SET) != st.st_size)
	    {
	      (void) __munmap (p, st.st_size);
	      fp->_offset = _IO_pos_BAD;
	    }
	  else
	    {
	      _IO_setb (fp, p, (char *) p + st.st_size, 0);

	      if (fp->_offset == _IO_pos_BAD)
		fp->_offset = 0;

	      _IO_setg (fp, p, p + fp->_offset, p + st.st_size);
	      fp->_offset = st.st_size;

	      if (fp->_mode <= 0)
		_IO_JUMPS_FILE_plus (fp) = &_IO_file_jumps_mmap;
	      else
		_IO_JUMPS_FILE_plus (fp) = &_IO_wfile_jumps_mmap;
	      fp->_wide_data->_wide_vtable = &_IO_wfile_jumps_mmap;

	      return;
	    }
	}
    }

  /* We couldn't use mmap, so revert to the vanilla file operations.  */

  if (fp->_mode <= 0)
    _IO_JUMPS_FILE_plus (fp) = &_IO_file_jumps;
  else
    _IO_JUMPS_FILE_plus (fp) = &_IO_wfile_jumps;
  fp->_wide_data->_wide_vtable = &_IO_wfile_jumps;
}

int
_IO_file_underflow_maybe_mmap (FILE *fp)
{
  /* This is the first read attempt.  Choose mmap or vanilla operations
     and then punt to the chosen underflow routine.  */
  decide_maybe_mmap (fp);
  return _IO_UNDERFLOW (fp);
}


int
_IO_new_file_overflow (FILE *f, int ch)
{
  if (f->_flags & _IO_NO_WRITES) /* SET ERROR */
    {
      f->_flags |= _IO_ERR_SEEN;
      __set_errno (EBADF);
      return EOF;
    }
  /* If currently reading or no buffer allocated. */
  if ((f->_flags & _IO_CURRENTLY_PUTTING) == 0 || f->_IO_write_base == NULL)
    {
      /* Allocate a buffer if needed. */
      if (f->_IO_write_base == NULL)
	{
	  _IO_doallocbuf (f);
	  _IO_setg (f, f->_IO_buf_base, f->_IO_buf_base, f->_IO_buf_base);
	}
      /* Otherwise must be currently reading.
	 If _IO_read_ptr (and hence also _IO_read_end) is at the buffer end,
	 logically slide the buffer forwards one block (by setting the
	 read pointers to all point at the beginning of the block).  This
	 makes room for subsequent output.
	 Otherwise, set the read pointers to _IO_read_end (leaving that
	 alone, so it can continue to correspond to the external position). */
      if (__glibc_unlikely (_IO_in_backup (f)))
	{
	  size_t nbackup = f->_IO_read_end - f->_IO_read_ptr;
	  _IO_free_backup_area (f);
	  f->_IO_read_base -= MIN (nbackup,
				   f->_IO_read_base - f->_IO_buf_base);
	  f->_IO_read_ptr = f->_IO_read_base;
	}

      if (f->_IO_read_ptr == f->_IO_buf_end)
	f->_IO_read_end = f->_IO_read_ptr = f->_IO_buf_base;
      f->_IO_write_ptr = f->_IO_read_ptr;
      f->_IO_write_base = f->_IO_write_ptr;
      f->_IO_write_end = f->_IO_buf_end;
      f->_IO_read_base = f->_IO_read_ptr = f->_IO_read_end;

      f->_flags |= _IO_CURRENTLY_PUTTING;
      if (f->_mode <= 0 && f->_flags & (_IO_LINE_BUF | _IO_UNBUFFERED))
	f->_IO_write_end = f->_IO_write_ptr;
    }
  if (ch == EOF)
    return _IO_do_write (f, f->_IO_write_base,
			 f->_IO_write_ptr - f->_IO_write_base);
  if (f->_IO_write_ptr == f->_IO_buf_end ) /* Buffer is really full */
    if (_IO_do_flush (f) == EOF)
      return EOF;
  *f->_IO_write_ptr++ = ch;
  if ((f->_flags & _IO_UNBUFFERED)
      || ((f->_flags & _IO_LINE_BUF) && ch == '\n'))
    if (_IO_do_write (f, f->_IO_write_base,
		      f->_IO_write_ptr - f->_IO_write_base) == EOF)
      return EOF;
  return (unsigned char) ch;
}
libc_hidden_ver (_IO_new_file_overflow, _IO_file_overflow)

int
_IO_new_file_sync (FILE *fp)
{
  ssize_t delta;
  int retval = 0;

  /*    char* ptr = cur_ptr(); */
  if (fp->_IO_write_ptr > fp->_IO_write_base)
    if (_IO_do_flush(fp)) return EOF;
  delta = fp->_IO_read_ptr - fp->_IO_read_end;
  if (delta != 0)
    {
      off64_t new_pos = _IO_SYSSEEK (fp, delta, 1);
      if (new_pos != (off64_t) EOF)
	fp->_IO_read_end = fp->_IO_read_ptr;
      else if (errno == ESPIPE)
	; /* Ignore error from unseekable devices. */
      else
	retval = EOF;
    }
  if (retval != EOF)
    fp->_offset = _IO_pos_BAD;
  /* FIXME: Cleanup - can this be shared? */
  /*    setg(base(), ptr, ptr); */
  return retval;
}
libc_hidden_ver (_IO_new_file_sync, _IO_file_sync)

static int
_IO_file_sync_mmap (FILE *fp)
{
  if (fp->_IO_read_ptr != fp->_IO_read_end)
    {
      if (__lseek64 (fp->_fileno, fp->_IO_read_ptr - fp->_IO_buf_base,
		     SEEK_SET)
	  != fp->_IO_read_ptr - fp->_IO_buf_base)
	{
	  fp->_flags |= _IO_ERR_SEEN;
	  return EOF;
	}
    }
  fp->_offset = fp->_IO_read_ptr - fp->_IO_buf_base;
  fp->_IO_read_end = fp->_IO_read_ptr = fp->_IO_read_base;
  return 0;
}

/* ftell{,o} implementation.  The only time we modify the state of the stream
   is when we have unflushed writes.  In that case we seek to the end and
   record that offset in the stream object.  */
static off64_t
do_ftell (FILE *fp)
{
  off64_t result, offset = 0;

  /* No point looking at unflushed data if we haven't allocated buffers
     yet.  */
  if (fp->_IO_buf_base != NULL)
    {
      bool unflushed_writes = fp->_IO_write_ptr > fp->_IO_write_base;

      bool append_mode = (fp->_flags & _IO_IS_APPENDING) == _IO_IS_APPENDING;

      /* When we have unflushed writes in append mode, seek to the end of the
	 file and record that offset.  This is the only time we change the file
	 stream state and it is safe since the file handle is active.  */
      if (unflushed_writes && append_mode)
	{
	  result = _IO_SYSSEEK (fp, 0, _IO_seek_end);
	  if (result == _IO_pos_BAD)
	    return EOF;
	  else
	    fp->_offset = result;
	}

      /* Adjust for unflushed data.  */
      if (!unflushed_writes)
	offset -= fp->_IO_read_end - fp->_IO_read_ptr;
      /* We don't trust _IO_read_end to represent the current file offset when
	 writing in append mode because the value would have to be shifted to
	 the end of the file during a flush.  Use the write base instead, along
	 with the new offset we got above when we did a seek to the end of the
	 file.  */
      else if (append_mode)
	offset += fp->_IO_write_ptr - fp->_IO_write_base;
      /* For all other modes, _IO_read_end represents the file offset.  */
      else
	offset += fp->_IO_write_ptr - fp->_IO_read_end;
    }

  if (fp->_offset != _IO_pos_BAD)
    result = fp->_offset;
  else
    result = _IO_SYSSEEK (fp, 0, _IO_seek_cur);

  if (result == EOF)
    return result;

  result += offset;

  if (result < 0)
    {
      __set_errno (EINVAL);
      return EOF;
    }

  return result;
}

off64_t
_IO_new_file_seekoff (FILE *fp, off64_t offset, int dir, int mode)
{
  off64_t result;
  off64_t delta, new_offset;
  long count;

  /* Short-circuit into a separate function.  We don't want to mix any
     functionality and we don't want to touch anything inside the FILE
     object. */
  if (mode == 0)
    return do_ftell (fp);

  /* POSIX.1 8.2.3.7 says that after a call the fflush() the file
     offset of the underlying file must be exact.  */
  int must_be_exact = (fp->_IO_read_base == fp->_IO_read_end
		       && fp->_IO_write_base == fp->_IO_write_ptr);

  bool was_writing = (fp->_IO_write_ptr > fp->_IO_write_base
		      || _IO_in_put_mode (fp));

  /* Flush unwritten characters.
     (This may do an unneeded write if we seek within the buffer.
     But to be able to switch to reading, we would need to set
     egptr to pptr.  That can't be done in the current design,
     which assumes file_ptr() is eGptr.  Anyway, since we probably
     end up flushing when we close(), it doesn't make much difference.)
     FIXME: simulate mem-mapped files. */
  if (was_writing && _IO_switch_to_get_mode (fp))
    return EOF;

  if (fp->_IO_buf_base == NULL)
    {
      /* It could be that we already have a pushback buffer.  */
      if (fp->_IO_read_base != NULL)
	{
	  free (fp->_IO_read_base);
	  fp->_flags &= ~_IO_IN_BACKUP;
	}
      _IO_doallocbuf (fp);
      _IO_setp (fp, fp->_IO_buf_base, fp->_IO_buf_base);
      _IO_setg (fp, fp->_IO_buf_base, fp->_IO_buf_base, fp->_IO_buf_base);
    }

  switch (dir)
    {
    case _IO_seek_cur:
      /* Adjust for read-ahead (bytes is buffer). */
      offset -= fp->_IO_read_end - fp->_IO_read_ptr;

      if (fp->_offset == _IO_pos_BAD)
	goto dumb;
      /* Make offset absolute, assuming current pointer is file_ptr(). */
      offset += fp->_offset;
      if (offset < 0)
	{
	  __set_errno (EINVAL);
	  return EOF;
	}

      dir = _IO_seek_set;
      break;
    case _IO_seek_set:
      break;
    case _IO_seek_end:
      {
	struct __stat64_t64 st;
	if (_IO_SYSSTAT (fp, &st) == 0 && S_ISREG (st.st_mode))
	  {
	    offset += st.st_size;
	    dir = _IO_seek_set;
	  }
	else
	  goto dumb;
      }
    }

  _IO_free_backup_area (fp);

  /* At this point, dir==_IO_seek_set. */

  /* If destination is within current buffer, optimize: */
  if (fp->_offset != _IO_pos_BAD && fp->_IO_read_base != NULL
      && !_IO_in_backup (fp))
    {
      off64_t start_offset = (fp->_offset
                              - (fp->_IO_read_end - fp->_IO_buf_base));
      if (offset >= start_offset && offset < fp->_offset)
	{
	  _IO_setg (fp, fp->_IO_buf_base,
		    fp->_IO_buf_base + (offset - start_offset),
		    fp->_IO_read_end);
	  _IO_setp (fp, fp->_IO_buf_base, fp->_IO_buf_base);

	  _IO_mask_flags (fp, 0, _IO_EOF_SEEN);
	  goto resync;
	}
    }

  if (fp->_flags & _IO_NO_READS)
    goto dumb;

  /* Try to seek to a block boundary, to improve kernel page management. */
  new_offset = offset & ~(fp->_IO_buf_end - fp->_IO_buf_base - 1);
  delta = offset - new_offset;
  if (delta > fp->_IO_buf_end - fp->_IO_buf_base)
    {
      new_offset = offset;
      delta = 0;
    }
  result = _IO_SYSSEEK (fp, new_offset, 0);
  if (result < 0)
    return EOF;
  if (delta == 0)
    count = 0;
  else
    {
      count = _IO_SYSREAD (fp, fp->_IO_buf_base,
			   (must_be_exact
			    ? delta : fp->_IO_buf_end - fp->_IO_buf_base));
      if (count < delta)
	{
	  /* We weren't allowed to read, but try to seek the remainder. */
	  offset = count == EOF ? delta : delta-count;
	  dir = _IO_seek_cur;
	  goto dumb;
	}
    }
  _IO_setg (fp, fp->_IO_buf_base, fp->_IO_buf_base + delta,
	    fp->_IO_buf_base + count);
  _IO_setp (fp, fp->_IO_buf_base, fp->_IO_buf_base);
  fp->_offset = result + count;
  _IO_mask_flags (fp, 0, _IO_EOF_SEEN);
  return offset;
 dumb:

  _IO_unsave_markers (fp);
  result = _IO_SYSSEEK (fp, offset, dir);
  if (result != EOF)
    {
      _IO_mask_flags (fp, 0, _IO_EOF_SEEN);
      fp->_offset = result;
      _IO_setg (fp, fp->_IO_buf_base, fp->_IO_buf_base, fp->_IO_buf_base);
      _IO_setp (fp, fp->_IO_buf_base, fp->_IO_buf_base);
    }
  return result;

resync:
  /* We need to do it since it is possible that the file offset in
     the kernel may be changed behind our back. It may happen when
     we fopen a file and then do a fork. One process may access the
     file and the kernel file offset will be changed. */
  if (fp->_offset >= 0)
    _IO_SYSSEEK (fp, fp->_offset, 0);

  return offset;
}
libc_hidden_ver (_IO_new_file_seekoff, _IO_file_seekoff)

off64_t
_IO_file_seekoff_mmap (FILE *fp, off64_t offset, int dir, int mode)
{
  off64_t result;

  /* If we are only interested in the current position, calculate it and
     return right now.  This calculation does the right thing when we are
     using a pushback buffer, but in the usual case has the same value as
     (fp->_IO_read_ptr - fp->_IO_buf_base).  */
  if (mode == 0)
    return fp->_offset - (fp->_IO_read_end - fp->_IO_read_ptr);

  switch (dir)
    {
    case _IO_seek_cur:
      /* Adjust for read-ahead (bytes is buffer). */
      offset += fp->_IO_read_ptr - fp->_IO_read_base;
      break;
    case _IO_seek_set:
      break;
    case _IO_seek_end:
      offset += fp->_IO_buf_end - fp->_IO_buf_base;
      break;
    }
  /* At this point, dir==_IO_seek_set. */

  if (offset < 0)
    {
      /* No negative offsets are valid.  */
      __set_errno (EINVAL);
      return EOF;
    }

  result = _IO_SYSSEEK (fp, offset, 0);
  if (result < 0)
    return EOF;

  if (offset > fp->_IO_buf_end - fp->_IO_buf_base)
    /* One can fseek arbitrarily past the end of the file
       and it is meaningless until one attempts to read.
       Leave the buffer pointers in EOF state until underflow.  */
    _IO_setg (fp, fp->_IO_buf_base, fp->_IO_buf_end, fp->_IO_buf_end);
  else
    /* Adjust the read pointers to match the file position,
       but so the next read attempt will call underflow.  */
    _IO_setg (fp, fp->_IO_buf_base, fp->_IO_buf_base + offset,
	      fp->_IO_buf_base + offset);

  fp->_offset = result;

  _IO_mask_flags (fp, 0, _IO_EOF_SEEN);

  return offset;
}

static off64_t
_IO_file_seekoff_maybe_mmap (FILE *fp, off64_t offset, int dir,
			     int mode)
{
  /* We only get here when we haven't tried to read anything yet.
     So there is nothing more useful for us to do here than just
     the underlying lseek call.  */

  off64_t result = _IO_SYSSEEK (fp, offset, dir);
  if (result < 0)
    return EOF;

  fp->_offset = result;
  return result;
}

ssize_t
_IO_file_read (FILE *fp, void *buf, ssize_t size)
{
  return (__builtin_expect (fp->_flags2 & _IO_FLAGS2_NOTCANCEL, 0)
	  ? __read_nocancel (fp->_fileno, buf, size)
	  : __read (fp->_fileno, buf, size));
}
libc_hidden_def (_IO_file_read)

off64_t
_IO_file_seek (FILE *fp, off64_t offset, int dir)
{
  return __lseek64 (fp->_fileno, offset, dir);
}
libc_hidden_def (_IO_file_seek)

int
_IO_file_stat (FILE *fp, void *st)
{
  return __fstat64_time64 (fp->_fileno, (struct __stat64_t64 *) st);
}
libc_hidden_def (_IO_file_stat)

int
_IO_file_close_mmap (FILE *fp)
{
  /* In addition to closing the file descriptor we have to unmap the file.  */
  (void) __munmap (fp->_IO_buf_base, fp->_IO_buf_end - fp->_IO_buf_base);
  fp->_IO_buf_base = fp->_IO_buf_end = NULL;
  /* Cancelling close should be avoided if possible since it leaves an
     unrecoverable state behind.  */
  return __close_nocancel (fp->_fileno);
}

int
_IO_file_close (FILE *fp)
{
  /* Cancelling close should be avoided if possible since it leaves an
     unrecoverable state behind.  */
  return __close_nocancel (fp->_fileno);
}
libc_hidden_def (_IO_file_close)

ssize_t
_IO_new_file_write (FILE *f, const void *data, ssize_t n)
{
  ssize_t to_do = n;
  while (to_do > 0)
    {
      ssize_t count = (__builtin_expect (f->_flags2
                                         & _IO_FLAGS2_NOTCANCEL, 0)
			   ? __write_nocancel (f->_fileno, data, to_do)
			   : __write (f->_fileno, data, to_do));
      if (count < 0)
	{
	  f->_flags |= _IO_ERR_SEEN;
	  break;
	}
      to_do -= count;
      data = (void *) ((char *) data + count);
    }
  n -= to_do;
  if (f->_offset >= 0)
    f->_offset += n;
  return n;
}

size_t
_IO_new_file_xsputn (FILE *f, const void *data, size_t n)
{
  const char *s = (const char *) data;
  size_t to_do = n;
  int must_flush = 0;
  size_t count = 0;

  if (n <= 0)
    return 0;
  /* This is an optimized implementation.
     If the amount to be written straddles a block boundary
     (or the filebuf is unbuffered), use sys_write directly. */

  /* First figure out how much space is available in the buffer. */
  if ((f->_flags & _IO_LINE_BUF) && (f->_flags & _IO_CURRENTLY_PUTTING))
    {
      count = f->_IO_buf_end - f->_IO_write_ptr;
      if (count >= n)
	{
	  const char *p;
	  for (p = s + n; p > s; )
	    {
	      if (*--p == '\n')
		{
		  count = p - s + 1;
		  must_flush = 1;
		  break;
		}
	    }
	}
    }
  else if (f->_IO_write_end > f->_IO_write_ptr)
    count = f->_IO_write_end - f->_IO_write_ptr; /* Space available. */

  /* Then fill the buffer. */
  if (count > 0)
    {
      if (count > to_do)
	count = to_do;
      f->_IO_write_ptr = __mempcpy (f->_IO_write_ptr, s, count);
      s += count;
      to_do -= count;
    }
  if (to_do + must_flush > 0)
    {
      size_t block_size, do_write;
      /* Next flush the (full) buffer. */
      if (_IO_OVERFLOW (f, EOF) == EOF)
	/* If nothing else has to be written we must not signal the
	   caller that everything has been written.  */
	return to_do == 0 ? EOF : n - to_do;

      /* Try to maintain alignment: write a whole number of blocks.  */
      block_size = f->_IO_buf_end - f->_IO_buf_base;
      do_write = to_do - (block_size >= 128 ? to_do % block_size : 0);

      if (do_write)
	{
	  count = new_do_write (f, s, do_write);
	  to_do -= count;
	  if (count < do_write)
	    return n - to_do;
	}

      /* Now write out the remainder.  Normally, this will fit in the
	 buffer, but it's somewhat messier for line-buffered files,
	 so we let _IO_default_xsputn handle the general case. */
      if (to_do)
	to_do -= _IO_default_xsputn (f, s+do_write, to_do);
    }
  return n - to_do;
}
libc_hidden_ver (_IO_new_file_xsputn, _IO_file_xsputn)

size_t
_IO_file_xsgetn (FILE *fp, void *data, size_t n)
{
  size_t want, have;
  ssize_t count;
  char *s = data;

  want = n;

  if (fp->_IO_buf_base == NULL)
    {
      /* Maybe we already have a push back pointer.  */
      if (fp->_IO_save_base != NULL)
	{
	  free (fp->_IO_save_base);
	  fp->_flags &= ~_IO_IN_BACKUP;
	}
      _IO_doallocbuf (fp);
    }

  while (want > 0)
    {
      have = fp->_IO_read_end - fp->_IO_read_ptr;
      if (want <= have)
	{
	  memcpy (s, fp->_IO_read_ptr, want);
	  fp->_IO_read_ptr += want;
	  want = 0;
	}
      else
	{
	  if (have > 0)
	    {
	      s = __mempcpy (s, fp->_IO_read_ptr, have);
	      want -= have;
	      fp->_IO_read_ptr += have;
	    }

	  /* Check for backup and repeat */
	  if (_IO_in_backup (fp))
	    {
	      _IO_switch_to_main_get_area (fp);
	      continue;
	    }

	  /* If we now want less than a buffer, underflow and repeat
	     the copy.  Otherwise, _IO_SYSREAD directly to
	     the user buffer. */
	  if (fp->_IO_buf_base
	      && want < (size_t) (fp->_IO_buf_end - fp->_IO_buf_base))
	    {
	      if (__underflow (fp) == EOF)
		break;

	      continue;
	    }

	  /* These must be set before the sysread as we might longjmp out
	     waiting for input. */
	  _IO_setg (fp, fp->_IO_buf_base, fp->_IO_buf_base, fp->_IO_buf_base);
	  _IO_setp (fp, fp->_IO_buf_base, fp->_IO_buf_base);

	  /* Try to maintain alignment: read a whole number of blocks.  */
	  count = want;
	  if (fp->_IO_buf_base)
	    {
	      size_t block_size = fp->_IO_buf_end - fp->_IO_buf_base;
	      if (block_size >= 128)
		count -= want % block_size;
	    }

	  count = _IO_SYSREAD (fp, s, count);
	  if (count <= 0)
	    {
	      if (count == 0)
		fp->_flags |= _IO_EOF_SEEN;
	      else
		fp->_flags |= _IO_ERR_SEEN;

	      break;
	    }

	  s += count;
	  want -= count;
	  if (fp->_offset != _IO_pos_BAD)
	    _IO_pos_adjust (fp->_offset, count);
	}
    }

  return n - want;
}
libc_hidden_def (_IO_file_xsgetn)

static size_t
_IO_file_xsgetn_mmap (FILE *fp, void *data, size_t n)
{
  size_t have;
  char *read_ptr = fp->_IO_read_ptr;
  char *s = (char *) data;

  have = fp->_IO_read_end - fp->_IO_read_ptr;

  if (have < n)
    {
      if (__glibc_unlikely (_IO_in_backup (fp)))
	{
	  s = __mempcpy (s, read_ptr, have);
	  n -= have;
	  _IO_switch_to_main_get_area (fp);
	  read_ptr = fp->_IO_read_ptr;
	  have = fp->_IO_read_end - fp->_IO_read_ptr;
	}

      if (have < n)
	{
	  /* Check that we are mapping all of the file, in case it grew.  */
	  if (__glibc_unlikely (mmap_remap_check (fp)))
	    /* We punted mmap, so complete with the vanilla code.  */
	    return s - (char *) data + _IO_XSGETN (fp, data, n);

	  read_ptr = fp->_IO_read_ptr;
	  have = fp->_IO_read_end - read_ptr;
	}
    }

  if (have < n)
    fp->_flags |= _IO_EOF_SEEN;

  if (have != 0)
    {
      have = MIN (have, n);
      s = __mempcpy (s, read_ptr, have);
      fp->_IO_read_ptr = read_ptr + have;
    }

  return s - (char *) data;
}

static size_t
_IO_file_xsgetn_maybe_mmap (FILE *fp, void *data, size_t n)
{
  /* We only get here if this is the first attempt to read something.
     Decide which operations to use and then punt to the chosen one.  */

  decide_maybe_mmap (fp);
  return _IO_XSGETN (fp, data, n);
}

versioned_symbol (libc, _IO_new_do_write, _IO_do_write, GLIBC_2_1);
versioned_symbol (libc, _IO_new_file_attach, _IO_file_attach, GLIBC_2_1);
versioned_symbol (libc, _IO_new_file_close_it, _IO_file_close_it, GLIBC_2_1);
versioned_symbol (libc, _IO_new_file_finish, _IO_file_finish, GLIBC_2_1);
versioned_symbol (libc, _IO_new_file_fopen, _IO_file_fopen, GLIBC_2_1);
versioned_symbol (libc, _IO_new_file_init, _IO_file_init, GLIBC_2_1);
versioned_symbol (libc, _IO_new_file_setbuf, _IO_file_setbuf, GLIBC_2_1);
versioned_symbol (libc, _IO_new_file_sync, _IO_file_sync, GLIBC_2_1);
versioned_symbol (libc, _IO_new_file_overflow, _IO_file_overflow, GLIBC_2_1);
versioned_symbol (libc, _IO_new_file_seekoff, _IO_file_seekoff, GLIBC_2_1);
versioned_symbol (libc, _IO_new_file_underflow, _IO_file_underflow, GLIBC_2_1);
versioned_symbol (libc, _IO_new_file_write, _IO_file_write, GLIBC_2_1);
versioned_symbol (libc, _IO_new_file_xsputn, _IO_file_xsputn, GLIBC_2_1);

const struct _IO_jump_t _IO_file_jumps libio_vtable =
{
  JUMP_INIT_DUMMY,
  JUMP_INIT(finish, _IO_file_finish),
  JUMP_INIT(overflow, _IO_file_overflow),
  JUMP_INIT(underflow, _IO_file_underflow),
  JUMP_INIT(uflow, _IO_default_uflow),
  JUMP_INIT(pbackfail, _IO_default_pbackfail),
  JUMP_INIT(xsputn, _IO_file_xsputn),
  JUMP_INIT(xsgetn, _IO_file_xsgetn),
  JUMP_INIT(seekoff, _IO_new_file_seekoff),
  JUMP_INIT(seekpos, _IO_default_seekpos),
  JUMP_INIT(setbuf, _IO_new_file_setbuf),
  JUMP_INIT(sync, _IO_new_file_sync),
  JUMP_INIT(doallocate, _IO_file_doallocate),
  JUMP_INIT(read, _IO_file_read),
  JUMP_INIT(write, _IO_new_file_write),
  JUMP_INIT(seek, _IO_file_seek),
  JUMP_INIT(close, _IO_file_close),
  JUMP_INIT(stat, _IO_file_stat),
  JUMP_INIT(showmanyc, _IO_default_showmanyc),
  JUMP_INIT(imbue, _IO_default_imbue)
};
libc_hidden_data_def (_IO_file_jumps)

const struct _IO_jump_t _IO_file_jumps_mmap libio_vtable =
{
  JUMP_INIT_DUMMY,
  JUMP_INIT(finish, _IO_file_finish),
  JUMP_INIT(overflow, _IO_file_overflow),
  JUMP_INIT(underflow, _IO_file_underflow_mmap),
  JUMP_INIT(uflow, _IO_default_uflow),
  JUMP_INIT(pbackfail, _IO_default_pbackfail),
  JUMP_INIT(xsputn, _IO_new_file_xsputn),
  JUMP_INIT(xsgetn, _IO_file_xsgetn_mmap),
  JUMP_INIT(seekoff, _IO_file_seekoff_mmap),
  JUMP_INIT(seekpos, _IO_default_seekpos),
  JUMP_INIT(setbuf, (_IO_setbuf_t) _IO_file_setbuf_mmap),
  JUMP_INIT(sync, _IO_file_sync_mmap),
  JUMP_INIT(doallocate, _IO_file_doallocate),
  JUMP_INIT(read, _IO_file_read),
  JUMP_INIT(write, _IO_new_file_write),
  JUMP_INIT(seek, _IO_file_seek),
  JUMP_INIT(close, _IO_file_close_mmap),
  JUMP_INIT(stat, _IO_file_stat),
  JUMP_INIT(showmanyc, _IO_default_showmanyc),
  JUMP_INIT(imbue, _IO_default_imbue)
};

const struct _IO_jump_t _IO_file_jumps_maybe_mmap libio_vtable =
{
  JUMP_INIT_DUMMY,
  JUMP_INIT(finish, _IO_file_finish),
  JUMP_INIT(overflow, _IO_file_overflow),
  JUMP_INIT(underflow, _IO_file_underflow_maybe_mmap),
  JUMP_INIT(uflow, _IO_default_uflow),
  JUMP_INIT(pbackfail, _IO_default_pbackfail),
  JUMP_INIT(xsputn, _IO_new_file_xsputn),
  JUMP_INIT(xsgetn, _IO_file_xsgetn_maybe_mmap),
  JUMP_INIT(seekoff, _IO_file_seekoff_maybe_mmap),
  JUMP_INIT(seekpos, _IO_default_seekpos),
  JUMP_INIT(setbuf, (_IO_setbuf_t) _IO_file_setbuf_mmap),
  JUMP_INIT(sync, _IO_new_file_sync),
  JUMP_INIT(doallocate, _IO_file_doallocate),
  JUMP_INIT(read, _IO_file_read),
  JUMP_INIT(write, _IO_new_file_write),
  JUMP_INIT(seek, _IO_file_seek),
  JUMP_INIT(close, _IO_file_close),
  JUMP_INIT(stat, _IO_file_stat),
  JUMP_INIT(showmanyc, _IO_default_showmanyc),
  JUMP_INIT(imbue, _IO_default_imbue)
};
