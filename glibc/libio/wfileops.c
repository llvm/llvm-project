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

#include <assert.h>
#include <libioP.h>
#include <wchar.h>
#include <gconv.h>
#include <stdlib.h>
#include <string.h>

/* Convert TO_DO wide character from DATA to FP.
   Then mark FP as having empty buffers. */
int
_IO_wdo_write (FILE *fp, const wchar_t *data, size_t to_do)
{
  struct _IO_codecvt *cc = fp->_codecvt;

  if (to_do > 0)
    {
      if (fp->_IO_write_end == fp->_IO_write_ptr
	  && fp->_IO_write_end != fp->_IO_write_base)
	{
	  if (_IO_new_do_write (fp, fp->_IO_write_base,
				fp->_IO_write_ptr - fp->_IO_write_base) == EOF)
	    return WEOF;
	}

      do
	{
	  enum __codecvt_result result;
	  const wchar_t *new_data;
	  char mb_buf[MB_LEN_MAX];
	  char *write_base, *write_ptr, *buf_end;

	  if (fp->_IO_write_ptr - fp->_IO_write_base < sizeof (mb_buf))
	    {
	      /* Make sure we have room for at least one multibyte
		 character.  */
	      write_ptr = write_base = mb_buf;
	      buf_end = mb_buf + sizeof (mb_buf);
	    }
	  else
	    {
	      write_ptr = fp->_IO_write_ptr;
	      write_base = fp->_IO_write_base;
	      buf_end = fp->_IO_buf_end;
	    }

	  /* Now convert from the internal format into the external buffer.  */
	  result = __libio_codecvt_out (cc, &fp->_wide_data->_IO_state,
					data, data + to_do, &new_data,
					write_ptr,
					buf_end,
					&write_ptr);

	  /* Write out what we produced so far.  */
	  if (_IO_new_do_write (fp, write_base, write_ptr - write_base) == EOF)
	    /* Something went wrong.  */
	    return WEOF;

	  to_do -= new_data - data;

	  /* Next see whether we had problems during the conversion.  If yes,
	     we cannot go on.  */
	  if (result != __codecvt_ok
	      && (result != __codecvt_partial || new_data - data == 0))
	    break;

	  data = new_data;
	}
      while (to_do > 0);
    }

  _IO_wsetg (fp, fp->_wide_data->_IO_buf_base, fp->_wide_data->_IO_buf_base,
	     fp->_wide_data->_IO_buf_base);
  fp->_wide_data->_IO_write_base = fp->_wide_data->_IO_write_ptr
    = fp->_wide_data->_IO_buf_base;
  fp->_wide_data->_IO_write_end = ((fp->_flags & (_IO_LINE_BUF | _IO_UNBUFFERED))
				   ? fp->_wide_data->_IO_buf_base
				   : fp->_wide_data->_IO_buf_end);

  return to_do == 0 ? 0 : WEOF;
}
libc_hidden_def (_IO_wdo_write)


wint_t
_IO_wfile_underflow (FILE *fp)
{
  struct _IO_codecvt *cd;
  enum __codecvt_result status;
  ssize_t count;

  /* C99 requires EOF to be "sticky".  */
  if (fp->_flags & _IO_EOF_SEEN)
    return WEOF;

  if (__glibc_unlikely (fp->_flags & _IO_NO_READS))
    {
      fp->_flags |= _IO_ERR_SEEN;
      __set_errno (EBADF);
      return WEOF;
    }
  if (fp->_wide_data->_IO_read_ptr < fp->_wide_data->_IO_read_end)
    return *fp->_wide_data->_IO_read_ptr;

  cd = fp->_codecvt;

  /* Maybe there is something left in the external buffer.  */
  if (fp->_IO_read_ptr < fp->_IO_read_end)
    {
      /* There is more in the external.  Convert it.  */
      const char *read_stop = (const char *) fp->_IO_read_ptr;

      fp->_wide_data->_IO_last_state = fp->_wide_data->_IO_state;
      fp->_wide_data->_IO_read_base = fp->_wide_data->_IO_read_ptr =
	fp->_wide_data->_IO_buf_base;
      status = __libio_codecvt_in (cd, &fp->_wide_data->_IO_state,
				   fp->_IO_read_ptr, fp->_IO_read_end,
				   &read_stop,
				   fp->_wide_data->_IO_read_ptr,
				   fp->_wide_data->_IO_buf_end,
				   &fp->_wide_data->_IO_read_end);

      fp->_IO_read_base = fp->_IO_read_ptr;
      fp->_IO_read_ptr = (char *) read_stop;

      /* If we managed to generate some text return the next character.  */
      if (fp->_wide_data->_IO_read_ptr < fp->_wide_data->_IO_read_end)
	return *fp->_wide_data->_IO_read_ptr;

      if (status == __codecvt_error)
	{
	  __set_errno (EILSEQ);
	  fp->_flags |= _IO_ERR_SEEN;
	  return WEOF;
	}

      /* Move the remaining content of the read buffer to the beginning.  */
      memmove (fp->_IO_buf_base, fp->_IO_read_ptr,
	       fp->_IO_read_end - fp->_IO_read_ptr);
      fp->_IO_read_end = (fp->_IO_buf_base
			  + (fp->_IO_read_end - fp->_IO_read_ptr));
      fp->_IO_read_base = fp->_IO_read_ptr = fp->_IO_buf_base;
    }
  else
    fp->_IO_read_base = fp->_IO_read_ptr = fp->_IO_read_end =
      fp->_IO_buf_base;

  if (fp->_IO_buf_base == NULL)
    {
      /* Maybe we already have a push back pointer.  */
      if (fp->_IO_save_base != NULL)
	{
	  free (fp->_IO_save_base);
	  fp->_flags &= ~_IO_IN_BACKUP;
	}
      _IO_doallocbuf (fp);

      fp->_IO_read_base = fp->_IO_read_ptr = fp->_IO_read_end =
	fp->_IO_buf_base;
    }

  fp->_IO_write_base = fp->_IO_write_ptr = fp->_IO_write_end =
    fp->_IO_buf_base;

  if (fp->_wide_data->_IO_buf_base == NULL)
    {
      /* Maybe we already have a push back pointer.  */
      if (fp->_wide_data->_IO_save_base != NULL)
	{
	  free (fp->_wide_data->_IO_save_base);
	  fp->_flags &= ~_IO_IN_BACKUP;
	}
      _IO_wdoallocbuf (fp);
    }

  /* FIXME This can/should be moved to genops ?? */
  if (fp->_flags & (_IO_LINE_BUF | _IO_UNBUFFERED))
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

  fp->_wide_data->_IO_read_base = fp->_wide_data->_IO_read_ptr =
    fp->_wide_data->_IO_buf_base;
  fp->_wide_data->_IO_read_end = fp->_wide_data->_IO_buf_base;
  fp->_wide_data->_IO_write_base = fp->_wide_data->_IO_write_ptr =
    fp->_wide_data->_IO_write_end = fp->_wide_data->_IO_buf_base;

  const char *read_ptr_copy;
  char accbuf[MB_LEN_MAX];
  size_t naccbuf = 0;
 again:
  count = _IO_SYSREAD (fp, fp->_IO_read_end,
		       fp->_IO_buf_end - fp->_IO_read_end);
  if (count <= 0)
    {
      if (count == 0 && naccbuf == 0)
	{
	  fp->_flags |= _IO_EOF_SEEN;
	  fp->_offset = _IO_pos_BAD;
	}
      else
	fp->_flags |= _IO_ERR_SEEN, count = 0;
    }
  fp->_IO_read_end += count;
  if (count == 0)
    {
      if (naccbuf != 0)
	/* There are some bytes in the external buffer but they don't
	   convert to anything.  */
	__set_errno (EILSEQ);
      return WEOF;
    }
  if (fp->_offset != _IO_pos_BAD)
    _IO_pos_adjust (fp->_offset, count);

  /* Now convert the read input.  */
  fp->_wide_data->_IO_last_state = fp->_wide_data->_IO_state;
  fp->_IO_read_base = fp->_IO_read_ptr;
  const char *from = fp->_IO_read_ptr;
  const char *to = fp->_IO_read_end;
  size_t to_copy = count;
  if (__glibc_unlikely (naccbuf != 0))
    {
      to_copy = MIN (sizeof (accbuf) - naccbuf, count);
      to = __mempcpy (&accbuf[naccbuf], from, to_copy);
      naccbuf += to_copy;
      from = accbuf;
    }
  status = __libio_codecvt_in (cd, &fp->_wide_data->_IO_state,
			       from, to, &read_ptr_copy,
			       fp->_wide_data->_IO_read_end,
			       fp->_wide_data->_IO_buf_end,
			       &fp->_wide_data->_IO_read_end);

  if (__glibc_unlikely (naccbuf != 0))
    fp->_IO_read_ptr += MAX (0, read_ptr_copy - &accbuf[naccbuf - to_copy]);
  else
    fp->_IO_read_ptr = (char *) read_ptr_copy;
  if (fp->_wide_data->_IO_read_end == fp->_wide_data->_IO_buf_base)
    {
      if (status == __codecvt_error)
	{
	out_eilseq:
	  __set_errno (EILSEQ);
	  fp->_flags |= _IO_ERR_SEEN;
	  return WEOF;
	}

      /* The read bytes make no complete character.  Try reading again.  */
      assert (status == __codecvt_partial);

      if (naccbuf == 0)
	{
	  if (fp->_IO_read_base < fp->_IO_read_ptr)
	    {
	      /* Partially used the buffer for some input data that
		 produces no output.  */
	      size_t avail = fp->_IO_read_end - fp->_IO_read_ptr;
	      memmove (fp->_IO_read_base, fp->_IO_read_ptr, avail);
	      fp->_IO_read_ptr = fp->_IO_read_base;
	      fp->_IO_read_end -= avail;
	      goto again;
	    }
	  naccbuf = fp->_IO_read_end - fp->_IO_read_ptr;
	  if (naccbuf >= sizeof (accbuf))
	    goto out_eilseq;

	  memcpy (accbuf, fp->_IO_read_ptr, naccbuf);
	}
      else
	{
	  size_t used = read_ptr_copy - accbuf;
	  if (used > 0)
	    {
	      memmove (accbuf, read_ptr_copy, naccbuf - used);
	      naccbuf -= used;
	    }

	  if (naccbuf == sizeof (accbuf))
	    goto out_eilseq;
	}

      fp->_IO_read_ptr = fp->_IO_read_end = fp->_IO_read_base;

      goto again;
    }

  return *fp->_wide_data->_IO_read_ptr;
}
libc_hidden_def (_IO_wfile_underflow)


static wint_t
_IO_wfile_underflow_mmap (FILE *fp)
{
  struct _IO_codecvt *cd;
  const char *read_stop;

  if (__glibc_unlikely (fp->_flags & _IO_NO_READS))
    {
      fp->_flags |= _IO_ERR_SEEN;
      __set_errno (EBADF);
      return WEOF;
    }
  if (fp->_wide_data->_IO_read_ptr < fp->_wide_data->_IO_read_end)
    return *fp->_wide_data->_IO_read_ptr;

  cd = fp->_codecvt;

  /* Maybe there is something left in the external buffer.  */
  if (fp->_IO_read_ptr >= fp->_IO_read_end
      /* No.  But maybe the read buffer is not fully set up.  */
      && _IO_file_underflow_mmap (fp) == EOF)
    /* Nothing available.  _IO_file_underflow_mmap has set the EOF or error
       flags as appropriate.  */
    return WEOF;

  /* There is more in the external.  Convert it.  */
  read_stop = (const char *) fp->_IO_read_ptr;

  if (fp->_wide_data->_IO_buf_base == NULL)
    {
      /* Maybe we already have a push back pointer.  */
      if (fp->_wide_data->_IO_save_base != NULL)
	{
	  free (fp->_wide_data->_IO_save_base);
	  fp->_flags &= ~_IO_IN_BACKUP;
	}
      _IO_wdoallocbuf (fp);
    }

  fp->_wide_data->_IO_last_state = fp->_wide_data->_IO_state;
  fp->_wide_data->_IO_read_base = fp->_wide_data->_IO_read_ptr =
    fp->_wide_data->_IO_buf_base;
  __libio_codecvt_in (cd, &fp->_wide_data->_IO_state,
		      fp->_IO_read_ptr, fp->_IO_read_end,
		      &read_stop,
		      fp->_wide_data->_IO_read_ptr,
		      fp->_wide_data->_IO_buf_end,
		      &fp->_wide_data->_IO_read_end);

  fp->_IO_read_ptr = (char *) read_stop;

  /* If we managed to generate some text return the next character.  */
  if (fp->_wide_data->_IO_read_ptr < fp->_wide_data->_IO_read_end)
    return *fp->_wide_data->_IO_read_ptr;

  /* There is some garbage at the end of the file.  */
  __set_errno (EILSEQ);
  fp->_flags |= _IO_ERR_SEEN;
  return WEOF;
}

static wint_t
_IO_wfile_underflow_maybe_mmap (FILE *fp)
{
  /* This is the first read attempt.  Doing the underflow will choose mmap
     or vanilla operations and then punt to the chosen underflow routine.
     Then we can punt to ours.  */
  if (_IO_file_underflow_maybe_mmap (fp) == EOF)
    return WEOF;

  return _IO_WUNDERFLOW (fp);
}


wint_t
_IO_wfile_overflow (FILE *f, wint_t wch)
{
  if (f->_flags & _IO_NO_WRITES) /* SET ERROR */
    {
      f->_flags |= _IO_ERR_SEEN;
      __set_errno (EBADF);
      return WEOF;
    }
  /* If currently reading or no buffer allocated. */
  if ((f->_flags & _IO_CURRENTLY_PUTTING) == 0)
    {
      /* Allocate a buffer if needed. */
      if (f->_wide_data->_IO_write_base == 0)
	{
	  _IO_wdoallocbuf (f);
	  _IO_free_wbackup_area (f);
	  _IO_wsetg (f, f->_wide_data->_IO_buf_base,
		     f->_wide_data->_IO_buf_base, f->_wide_data->_IO_buf_base);

	  if (f->_IO_write_base == NULL)
	    {
	      _IO_doallocbuf (f);
	      _IO_setg (f, f->_IO_buf_base, f->_IO_buf_base, f->_IO_buf_base);
	    }
	}
      else
	{
	  /* Otherwise must be currently reading.  If _IO_read_ptr
	     (and hence also _IO_read_end) is at the buffer end,
	     logically slide the buffer forwards one block (by setting
	     the read pointers to all point at the beginning of the
	     block).  This makes room for subsequent output.
	     Otherwise, set the read pointers to _IO_read_end (leaving
	     that alone, so it can continue to correspond to the
	     external position). */
	  if (f->_wide_data->_IO_read_ptr == f->_wide_data->_IO_buf_end)
	    {
	      f->_IO_read_end = f->_IO_read_ptr = f->_IO_buf_base;
	      f->_wide_data->_IO_read_end = f->_wide_data->_IO_read_ptr =
		f->_wide_data->_IO_buf_base;
	    }
	}
      f->_wide_data->_IO_write_ptr = f->_wide_data->_IO_read_ptr;
      f->_wide_data->_IO_write_base = f->_wide_data->_IO_write_ptr;
      f->_wide_data->_IO_write_end = f->_wide_data->_IO_buf_end;
      f->_wide_data->_IO_read_base = f->_wide_data->_IO_read_ptr =
	f->_wide_data->_IO_read_end;

      f->_IO_write_ptr = f->_IO_read_ptr;
      f->_IO_write_base = f->_IO_write_ptr;
      f->_IO_write_end = f->_IO_buf_end;
      f->_IO_read_base = f->_IO_read_ptr = f->_IO_read_end;

      f->_flags |= _IO_CURRENTLY_PUTTING;
      if (f->_flags & (_IO_LINE_BUF | _IO_UNBUFFERED))
	f->_wide_data->_IO_write_end = f->_wide_data->_IO_write_ptr;
    }
  if (wch == WEOF)
    return _IO_do_flush (f);
  if (f->_wide_data->_IO_write_ptr == f->_wide_data->_IO_buf_end)
    /* Buffer is really full */
    if (_IO_do_flush (f) == EOF)
      return WEOF;
  *f->_wide_data->_IO_write_ptr++ = wch;
  if ((f->_flags & _IO_UNBUFFERED)
      || ((f->_flags & _IO_LINE_BUF) && wch == L'\n'))
    if (_IO_do_flush (f) == EOF)
      return WEOF;
  return wch;
}
libc_hidden_def (_IO_wfile_overflow)

wint_t
_IO_wfile_sync (FILE *fp)
{
  ssize_t delta;
  wint_t retval = 0;

  /*    char* ptr = cur_ptr(); */
  if (fp->_wide_data->_IO_write_ptr > fp->_wide_data->_IO_write_base)
    if (_IO_do_flush (fp))
      return WEOF;
  delta = fp->_wide_data->_IO_read_ptr - fp->_wide_data->_IO_read_end;
  if (delta != 0)
    {
      /* We have to find out how many bytes we have to go back in the
	 external buffer.  */
      struct _IO_codecvt *cv = fp->_codecvt;
      off64_t new_pos;

      int clen = __libio_codecvt_encoding (cv);

      if (clen > 0)
	/* It is easy, a fixed number of input bytes are used for each
	   wide character.  */
	delta *= clen;
      else
	{
	  /* We have to find out the hard way how much to back off.
	     To do this we determine how much input we needed to
	     generate the wide characters up to the current reading
	     position.  */
	  int nread;
	  size_t wnread = (fp->_wide_data->_IO_read_ptr
			   - fp->_wide_data->_IO_read_base);
	  fp->_wide_data->_IO_state = fp->_wide_data->_IO_last_state;
	  nread = __libio_codecvt_length (cv, &fp->_wide_data->_IO_state,
					  fp->_IO_read_base,
					  fp->_IO_read_end, wnread);
	  fp->_IO_read_ptr = fp->_IO_read_base + nread;
	  delta = -(fp->_IO_read_end - fp->_IO_read_base - nread);
	}

      new_pos = _IO_SYSSEEK (fp, delta, 1);
      if (new_pos != (off64_t) EOF)
	{
	  fp->_wide_data->_IO_read_end = fp->_wide_data->_IO_read_ptr;
	  fp->_IO_read_end = fp->_IO_read_ptr;
	}
      else if (errno == ESPIPE)
	; /* Ignore error from unseekable devices. */
      else
	retval = WEOF;
    }
  if (retval != WEOF)
    fp->_offset = _IO_pos_BAD;
  /* FIXME: Cleanup - can this be shared? */
  /*    setg(base(), ptr, ptr); */
  return retval;
}
libc_hidden_def (_IO_wfile_sync)

/* Adjust the internal buffer pointers to reflect the state in the external
   buffer.  The content between fp->_IO_read_base and fp->_IO_read_ptr is
   assumed to be converted and available in the range
   fp->_wide_data->_IO_read_base and fp->_wide_data->_IO_read_end.

   Returns 0 on success and -1 on error with the _IO_ERR_SEEN flag set.  */
static int
adjust_wide_data (FILE *fp, bool do_convert)
{
  struct _IO_codecvt *cv = fp->_codecvt;

  int clen = __libio_codecvt_encoding (cv);

  /* Take the easy way out for constant length encodings if we don't need to
     convert.  */
  if (!do_convert && clen > 0)
    {
      fp->_wide_data->_IO_read_end += ((fp->_IO_read_ptr - fp->_IO_read_base)
				       / clen);
      goto done;
    }

  enum __codecvt_result status;
  const char *read_stop = (const char *) fp->_IO_read_base;
  do
    {

      fp->_wide_data->_IO_last_state = fp->_wide_data->_IO_state;
      status = __libio_codecvt_in (cv, &fp->_wide_data->_IO_state,
				   fp->_IO_read_base, fp->_IO_read_ptr,
				   &read_stop,
				   fp->_wide_data->_IO_read_base,
				   fp->_wide_data->_IO_buf_end,
				   &fp->_wide_data->_IO_read_end);

      /* Should we return EILSEQ?  */
      if (__glibc_unlikely (status == __codecvt_error))
	{
	  fp->_flags |= _IO_ERR_SEEN;
	  return -1;
	}
    }
  while (__builtin_expect (status == __codecvt_partial, 0));

done:
  /* Now seek to _IO_read_end to behave as if we have read it all in.  */
  fp->_wide_data->_IO_read_ptr = fp->_wide_data->_IO_read_end;

  return 0;
}

/* ftell{,o} implementation for wide mode.  Don't modify any state of the file
   pointer while we try to get the current state of the stream except in one
   case, which is when we have unflushed writes in append mode.  */
static off64_t
do_ftell_wide (FILE *fp)
{
  off64_t result, offset = 0;

  /* No point looking for offsets in the buffer if it hasn't even been
     allocated.  */
  if (fp->_wide_data->_IO_buf_base != NULL)
    {
      const wchar_t *wide_read_base;
      const wchar_t *wide_read_ptr;
      const wchar_t *wide_read_end;
      bool unflushed_writes = (fp->_wide_data->_IO_write_ptr
			       > fp->_wide_data->_IO_write_base);

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

      /* XXX For wide stream with backup store it is not very
	 reasonable to determine the offset.  The pushed-back
	 character might require a state change and we need not be
	 able to compute the initial state by reverse transformation
	 since there is no guarantee of symmetry.  So we don't even
	 try and return an error.  */
      if (_IO_in_backup (fp))
	{
	  if (fp->_wide_data->_IO_read_ptr < fp->_wide_data->_IO_read_end)
	    {
	      __set_errno (EINVAL);
	      return -1;
	    }

	  /* Nothing in the backup store, so note the backed up pointers
	     without changing the state.  */
	  wide_read_base = fp->_wide_data->_IO_save_base;
	  wide_read_ptr = wide_read_base;
	  wide_read_end = fp->_wide_data->_IO_save_end;
	}
      else
	{
	  wide_read_base = fp->_wide_data->_IO_read_base;
	  wide_read_ptr = fp->_wide_data->_IO_read_ptr;
	  wide_read_end = fp->_wide_data->_IO_read_end;
	}

      struct _IO_codecvt *cv = fp->_codecvt;
      int clen = __libio_codecvt_encoding (cv);

      if (!unflushed_writes)
	{
	  if (clen > 0)
	    {
	      offset -= (wide_read_end - wide_read_ptr) * clen;
	      offset -= fp->_IO_read_end - fp->_IO_read_ptr;
	    }
	  else
	    {
	      int nread;

	      size_t delta = wide_read_ptr - wide_read_base;
	      __mbstate_t state = fp->_wide_data->_IO_last_state;
	      nread = __libio_codecvt_length (cv, &state,
					      fp->_IO_read_base,
					      fp->_IO_read_end, delta);
	      offset -= fp->_IO_read_end - fp->_IO_read_base - nread;
	    }
	}
      else
	{
	  if (clen > 0)
	    offset += (fp->_wide_data->_IO_write_ptr
		       - fp->_wide_data->_IO_write_base) * clen;
	  else
	    {
	      size_t delta = (fp->_wide_data->_IO_write_ptr
			      - fp->_wide_data->_IO_write_base);

	      /* Allocate enough space for the conversion.  */
	      size_t outsize = delta * sizeof (wchar_t);
	      char *out = malloc (outsize);
	      char *outstop = out;
	      const wchar_t *in = fp->_wide_data->_IO_write_base;

	      enum __codecvt_result status;

	      __mbstate_t state = fp->_wide_data->_IO_last_state;
	      status = __libio_codecvt_out (cv, &state, in, in + delta, &in,
					    out, out + outsize, &outstop);

	      /* We don't check for __codecvt_partial because it can be
		 returned on one of two conditions: either the output
		 buffer is full or the input sequence is incomplete.  We
		 take care to allocate enough buffer and our input
		 sequences must be complete since they are accepted as
		 wchar_t; if not, then that is an error.  */
	      if (__glibc_unlikely (status != __codecvt_ok))
		{
		  free (out);
		  return WEOF;
		}

	      offset += outstop - out;
	      free (out);
	    }

	  /* We don't trust _IO_read_end to represent the current file offset
	     when writing in append mode because the value would have to be
	     shifted to the end of the file during a flush.  Use the write base
	     instead, along with the new offset we got above when we did a seek
	     to the end of the file.  */
	  if (append_mode)
	    offset += fp->_IO_write_ptr - fp->_IO_write_base;
	  /* For all other modes, _IO_read_end represents the file offset.  */
	  else
	    offset += fp->_IO_write_ptr - fp->_IO_read_end;
	}
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
_IO_wfile_seekoff (FILE *fp, off64_t offset, int dir, int mode)
{
  off64_t result;
  off64_t delta, new_offset;
  long int count;

  /* Short-circuit into a separate function.  We don't want to mix any
     functionality and we don't want to touch anything inside the FILE
     object. */
  if (mode == 0)
    return do_ftell_wide (fp);

  /* POSIX.1 8.2.3.7 says that after a call the fflush() the file
     offset of the underlying file must be exact.  */
  int must_be_exact = ((fp->_wide_data->_IO_read_base
			== fp->_wide_data->_IO_read_end)
		       && (fp->_wide_data->_IO_write_base
			   == fp->_wide_data->_IO_write_ptr));

  bool was_writing = ((fp->_wide_data->_IO_write_ptr
		       > fp->_wide_data->_IO_write_base)
		      || _IO_in_put_mode (fp));

  /* Flush unwritten characters.
     (This may do an unneeded write if we seek within the buffer.
     But to be able to switch to reading, we would need to set
     egptr to pptr.  That can't be done in the current design,
     which assumes file_ptr() is eGptr.  Anyway, since we probably
     end up flushing when we close(), it doesn't make much difference.)
     FIXME: simulate mem-mapped files. */
  if (was_writing && _IO_switch_to_wget_mode (fp))
    return WEOF;

  if (fp->_wide_data->_IO_buf_base == NULL)
    {
      /* It could be that we already have a pushback buffer.  */
      if (fp->_wide_data->_IO_read_base != NULL)
	{
	  free (fp->_wide_data->_IO_read_base);
	  fp->_flags &= ~_IO_IN_BACKUP;
	}
      _IO_doallocbuf (fp);
      _IO_setp (fp, fp->_IO_buf_base, fp->_IO_buf_base);
      _IO_setg (fp, fp->_IO_buf_base, fp->_IO_buf_base, fp->_IO_buf_base);
      _IO_wsetp (fp, fp->_wide_data->_IO_buf_base,
		 fp->_wide_data->_IO_buf_base);
      _IO_wsetg (fp, fp->_wide_data->_IO_buf_base,
		 fp->_wide_data->_IO_buf_base, fp->_wide_data->_IO_buf_base);
    }

  switch (dir)
    {
      struct _IO_codecvt *cv;
      int clen;

    case _IO_seek_cur:
      /* Adjust for read-ahead (bytes is buffer).  To do this we must
	 find out which position in the external buffer corresponds to
	 the current position in the internal buffer.  */
      cv = fp->_codecvt;
      clen = __libio_codecvt_encoding (cv);

      if (mode != 0 || !was_writing)
	{
	  if (clen > 0)
	    {
	      offset -= (fp->_wide_data->_IO_read_end
			 - fp->_wide_data->_IO_read_ptr) * clen;
	      /* Adjust by readahead in external buffer.  */
	      offset -= fp->_IO_read_end - fp->_IO_read_ptr;
	    }
	  else
	    {
	      int nread;

	      delta = (fp->_wide_data->_IO_read_ptr
		       - fp->_wide_data->_IO_read_base);
	      fp->_wide_data->_IO_state = fp->_wide_data->_IO_last_state;
	      nread = __libio_codecvt_length (cv,
					      &fp->_wide_data->_IO_state,
					      fp->_IO_read_base,
					      fp->_IO_read_end, delta);
	      fp->_IO_read_ptr = fp->_IO_read_base + nread;
	      fp->_wide_data->_IO_read_end = fp->_wide_data->_IO_read_ptr;
	      offset -= fp->_IO_read_end - fp->_IO_read_base - nread;
	    }
	}

      if (fp->_offset == _IO_pos_BAD)
	goto dumb;

      /* Make offset absolute, assuming current pointer is file_ptr(). */
      offset += fp->_offset;

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

  _IO_free_wbackup_area (fp);

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
	  _IO_wsetg (fp, fp->_wide_data->_IO_buf_base,
		     fp->_wide_data->_IO_buf_base,
		     fp->_wide_data->_IO_buf_base);
	  _IO_wsetp (fp, fp->_wide_data->_IO_buf_base,
		     fp->_wide_data->_IO_buf_base);

	  if (adjust_wide_data (fp, false))
	    goto dumb;

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
  _IO_wsetg (fp, fp->_wide_data->_IO_buf_base,
	     fp->_wide_data->_IO_buf_base, fp->_wide_data->_IO_buf_base);
  _IO_wsetp (fp, fp->_wide_data->_IO_buf_base, fp->_wide_data->_IO_buf_base);

  if (adjust_wide_data (fp, true))
    goto dumb;

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
      _IO_wsetg (fp, fp->_wide_data->_IO_buf_base,
		 fp->_wide_data->_IO_buf_base, fp->_wide_data->_IO_buf_base);
      _IO_wsetp (fp, fp->_wide_data->_IO_buf_base,
		 fp->_wide_data->_IO_buf_base);
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
libc_hidden_def (_IO_wfile_seekoff)


size_t
_IO_wfile_xsputn (FILE *f, const void *data, size_t n)
{
  const wchar_t *s = (const wchar_t *) data;
  size_t to_do = n;
  int must_flush = 0;
  size_t count;

  if (n <= 0)
    return 0;
  /* This is an optimized implementation.
     If the amount to be written straddles a block boundary
     (or the filebuf is unbuffered), use sys_write directly. */

  /* First figure out how much space is available in the buffer. */
  count = f->_wide_data->_IO_write_end - f->_wide_data->_IO_write_ptr;
  if ((f->_flags & _IO_LINE_BUF) && (f->_flags & _IO_CURRENTLY_PUTTING))
    {
      count = f->_wide_data->_IO_buf_end - f->_wide_data->_IO_write_ptr;
      if (count >= n)
	{
	  const wchar_t *p;
	  for (p = s + n; p > s; )
	    {
	      if (*--p == L'\n')
		{
		  count = p - s + 1;
		  must_flush = 1;
		  break;
		}
	    }
	}
    }
  /* Then fill the buffer. */
  if (count > 0)
    {
      if (count > to_do)
	count = to_do;
      if (count > 20)
	{
	  f->_wide_data->_IO_write_ptr =
	    __wmempcpy (f->_wide_data->_IO_write_ptr, s, count);
	  s += count;
	}
      else
	{
	  wchar_t *p = f->_wide_data->_IO_write_ptr;
	  int i = (int) count;
	  while (--i >= 0)
	    *p++ = *s++;
	  f->_wide_data->_IO_write_ptr = p;
	}
      to_do -= count;
    }
  if (to_do > 0)
    to_do -= _IO_wdefault_xsputn (f, s, to_do);
  if (must_flush
      && f->_wide_data->_IO_write_ptr != f->_wide_data->_IO_write_base)
    _IO_wdo_write (f, f->_wide_data->_IO_write_base,
		   f->_wide_data->_IO_write_ptr
		   - f->_wide_data->_IO_write_base);

  return n - to_do;
}
libc_hidden_def (_IO_wfile_xsputn)


const struct _IO_jump_t _IO_wfile_jumps libio_vtable =
{
  JUMP_INIT_DUMMY,
  JUMP_INIT(finish, _IO_new_file_finish),
  JUMP_INIT(overflow, (_IO_overflow_t) _IO_wfile_overflow),
  JUMP_INIT(underflow, (_IO_underflow_t) _IO_wfile_underflow),
  JUMP_INIT(uflow, (_IO_underflow_t) _IO_wdefault_uflow),
  JUMP_INIT(pbackfail, (_IO_pbackfail_t) _IO_wdefault_pbackfail),
  JUMP_INIT(xsputn, _IO_wfile_xsputn),
  JUMP_INIT(xsgetn, _IO_file_xsgetn),
  JUMP_INIT(seekoff, _IO_wfile_seekoff),
  JUMP_INIT(seekpos, _IO_default_seekpos),
  JUMP_INIT(setbuf, _IO_new_file_setbuf),
  JUMP_INIT(sync, (_IO_sync_t) _IO_wfile_sync),
  JUMP_INIT(doallocate, _IO_wfile_doallocate),
  JUMP_INIT(read, _IO_file_read),
  JUMP_INIT(write, _IO_new_file_write),
  JUMP_INIT(seek, _IO_file_seek),
  JUMP_INIT(close, _IO_file_close),
  JUMP_INIT(stat, _IO_file_stat),
  JUMP_INIT(showmanyc, _IO_default_showmanyc),
  JUMP_INIT(imbue, _IO_default_imbue)
};
libc_hidden_data_def (_IO_wfile_jumps)


const struct _IO_jump_t _IO_wfile_jumps_mmap libio_vtable =
{
  JUMP_INIT_DUMMY,
  JUMP_INIT(finish, _IO_new_file_finish),
  JUMP_INIT(overflow, (_IO_overflow_t) _IO_wfile_overflow),
  JUMP_INIT(underflow, (_IO_underflow_t) _IO_wfile_underflow_mmap),
  JUMP_INIT(uflow, (_IO_underflow_t) _IO_wdefault_uflow),
  JUMP_INIT(pbackfail, (_IO_pbackfail_t) _IO_wdefault_pbackfail),
  JUMP_INIT(xsputn, _IO_wfile_xsputn),
  JUMP_INIT(xsgetn, _IO_file_xsgetn),
  JUMP_INIT(seekoff, _IO_wfile_seekoff),
  JUMP_INIT(seekpos, _IO_default_seekpos),
  JUMP_INIT(setbuf, _IO_file_setbuf_mmap),
  JUMP_INIT(sync, (_IO_sync_t) _IO_wfile_sync),
  JUMP_INIT(doallocate, _IO_wfile_doallocate),
  JUMP_INIT(read, _IO_file_read),
  JUMP_INIT(write, _IO_new_file_write),
  JUMP_INIT(seek, _IO_file_seek),
  JUMP_INIT(close, _IO_file_close_mmap),
  JUMP_INIT(stat, _IO_file_stat),
  JUMP_INIT(showmanyc, _IO_default_showmanyc),
  JUMP_INIT(imbue, _IO_default_imbue)
};

const struct _IO_jump_t _IO_wfile_jumps_maybe_mmap libio_vtable =
{
  JUMP_INIT_DUMMY,
  JUMP_INIT(finish, _IO_new_file_finish),
  JUMP_INIT(overflow, (_IO_overflow_t) _IO_wfile_overflow),
  JUMP_INIT(underflow, (_IO_underflow_t) _IO_wfile_underflow_maybe_mmap),
  JUMP_INIT(uflow, (_IO_underflow_t) _IO_wdefault_uflow),
  JUMP_INIT(pbackfail, (_IO_pbackfail_t) _IO_wdefault_pbackfail),
  JUMP_INIT(xsputn, _IO_wfile_xsputn),
  JUMP_INIT(xsgetn, _IO_file_xsgetn),
  JUMP_INIT(seekoff, _IO_wfile_seekoff),
  JUMP_INIT(seekpos, _IO_default_seekpos),
  JUMP_INIT(setbuf, _IO_file_setbuf_mmap),
  JUMP_INIT(sync, (_IO_sync_t) _IO_wfile_sync),
  JUMP_INIT(doallocate, _IO_wfile_doallocate),
  JUMP_INIT(read, _IO_file_read),
  JUMP_INIT(write, _IO_new_file_write),
  JUMP_INIT(seek, _IO_file_seek),
  JUMP_INIT(close, _IO_file_close),
  JUMP_INIT(stat, _IO_file_stat),
  JUMP_INIT(showmanyc, _IO_default_showmanyc),
  JUMP_INIT(imbue, _IO_default_imbue)
};
