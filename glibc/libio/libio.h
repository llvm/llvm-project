/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#ifndef _LIBIO_H
#define _LIBIO_H 1

#ifndef _LIBC
# error "libio.h should only be included when building glibc itself"
#endif
#ifdef _ISOMAC
# error "libio.h should not be included under _ISOMAC"
#endif

#include <stdio.h>

#if defined _IO_MTSAFE_IO && !defined _IO_lock_t_defined
# error "Someone forgot to include stdio-lock.h"
#endif

#define __need_wchar_t
#include <stddef.h>

#include <bits/types/__mbstate_t.h>
#include <bits/types/wint_t.h>
#include <gconv.h>

typedef struct
{
  struct __gconv_step *step;
  struct __gconv_step_data step_data;
} _IO_iconv_t;

#include <shlib-compat.h>

/* _IO_seekoff modes */
#define _IOS_INPUT	1
#define _IOS_OUTPUT	2

/* Magic number and bits for the _flags field.  The magic number is
   mostly vestigial, but preserved for compatibility.  It occupies the
   high 16 bits of _flags; the low 16 bits are actual flag bits.  */

#define _IO_MAGIC         0xFBAD0000 /* Magic number */
#define _IO_MAGIC_MASK    0xFFFF0000
#define _IO_USER_BUF          0x0001 /* Don't deallocate buffer on close. */
#define _IO_UNBUFFERED        0x0002
#define _IO_NO_READS          0x0004 /* Reading not allowed.  */
#define _IO_NO_WRITES         0x0008 /* Writing not allowed.  */
#define _IO_EOF_SEEN          0x0010
#define _IO_ERR_SEEN          0x0020
#define _IO_DELETE_DONT_CLOSE 0x0040 /* Don't call close(_fileno) on close.  */
#define _IO_LINKED            0x0080 /* In the list of all open files.  */
#define _IO_IN_BACKUP         0x0100
#define _IO_LINE_BUF          0x0200
#define _IO_TIED_PUT_GET      0x0400 /* Put and get pointer move in unison.  */
#define _IO_CURRENTLY_PUTTING 0x0800
#define _IO_IS_APPENDING      0x1000
#define _IO_IS_FILEBUF        0x2000
                           /* 0x4000  No longer used, reserved for compat.  */
#define _IO_USER_LOCK         0x8000

/* Bits for the _flags2 field.  */
#define _IO_FLAGS2_MMAP 1
#define _IO_FLAGS2_NOTCANCEL 2
#define _IO_FLAGS2_USER_WBUF 8
#define _IO_FLAGS2_NOCLOSE 32
#define _IO_FLAGS2_CLOEXEC 64
#define _IO_FLAGS2_NEED_LOCK 128

/* _IO_pos_BAD is an off64_t value indicating error, unknown, or EOF.  */
#define _IO_pos_BAD ((off64_t) -1)

/* _IO_pos_adjust adjusts an off64_t by some number of bytes.  */
#define _IO_pos_adjust(pos, delta) ((pos) += (delta))

/* _IO_pos_0 is an off64_t value indicating beginning of file.  */
#define _IO_pos_0 ((off64_t) 0)

struct _IO_jump_t;

/* A streammarker remembers a position in a buffer. */
struct _IO_marker {
  struct _IO_marker *_next;
  FILE *_sbuf;
  /* If _pos >= 0
 it points to _buf->Gbase()+_pos. FIXME comment */
  /* if _pos < 0, it points to _buf->eBptr()+_pos. FIXME comment */
  int _pos;
};

struct _IO_codecvt
{
  _IO_iconv_t __cd_in;
  _IO_iconv_t __cd_out;
};

/* Extra data for wide character streams.  */
struct _IO_wide_data
{
  wchar_t *_IO_read_ptr;	/* Current read pointer */
  wchar_t *_IO_read_end;	/* End of get area. */
  wchar_t *_IO_read_base;	/* Start of putback+get area. */
  wchar_t *_IO_write_base;	/* Start of put area. */
  wchar_t *_IO_write_ptr;	/* Current put pointer. */
  wchar_t *_IO_write_end;	/* End of put area. */
  wchar_t *_IO_buf_base;	/* Start of reserve area. */
  wchar_t *_IO_buf_end;		/* End of reserve area. */
  /* The following fields are used to support backing up and undo. */
  wchar_t *_IO_save_base;	/* Pointer to start of non-current get area. */
  wchar_t *_IO_backup_base;	/* Pointer to first valid character of
				   backup area */
  wchar_t *_IO_save_end;	/* Pointer to end of non-current get area. */

  __mbstate_t _IO_state;
  __mbstate_t _IO_last_state;
  struct _IO_codecvt _codecvt;

  wchar_t _shortbuf[1];

  const struct _IO_jump_t *_wide_vtable;
};

struct _IO_FILE_plus;

extern struct _IO_FILE_plus _IO_2_1_stdin_;
extern struct _IO_FILE_plus _IO_2_1_stdout_;
extern struct _IO_FILE_plus _IO_2_1_stderr_;

struct _IO_cookie_file;

/* Initialize one of those.  */
extern void _IO_cookie_init (struct _IO_cookie_file *__cfile, int __read_write,
			     void *__cookie, cookie_io_functions_t __fns);

extern int __underflow (FILE *);
extern wint_t __wunderflow (FILE *);
extern wint_t __wuflow (FILE *);
extern wint_t __woverflow (FILE *, wint_t);

#define _IO_getc_unlocked(_fp) __getc_unlocked_body (_fp)
#define _IO_peekc_unlocked(_fp)						\
  (__glibc_unlikely ((_fp)->_IO_read_ptr >= (_fp)->_IO_read_end)	\
   && __underflow (_fp) == EOF						\
   ? EOF								\
   : *(unsigned char *) (_fp)->_IO_read_ptr)
#define _IO_putc_unlocked(_ch, _fp) __putc_unlocked_body (_ch, _fp)

# define _IO_getwc_unlocked(_fp)					\
  (__glibc_unlikely ((_fp)->_wide_data == NULL				\
		     || ((_fp)->_wide_data->_IO_read_ptr		\
			 >= (_fp)->_wide_data->_IO_read_end))		\
   ? __wuflow (_fp) : (wint_t) *(_fp)->_wide_data->_IO_read_ptr++)
# define _IO_putwc_unlocked(_wch, _fp)					\
  (__glibc_unlikely ((_fp)->_wide_data == NULL				\
		     || ((_fp)->_wide_data->_IO_write_ptr		\
			 >= (_fp)->_wide_data->_IO_write_end))		\
   ? __woverflow (_fp, _wch)						\
   : (wint_t) (*(_fp)->_wide_data->_IO_write_ptr++ = (_wch)))

#define _IO_feof_unlocked(_fp) __feof_unlocked_body (_fp)
#define _IO_ferror_unlocked(_fp) __ferror_unlocked_body (_fp)

extern int _IO_getc (FILE *__fp);
extern int _IO_putc (int __c, FILE *__fp);
extern int _IO_feof (FILE *__fp) __THROW;
extern int _IO_ferror (FILE *__fp) __THROW;

extern int _IO_peekc_locked (FILE *__fp);

/* This one is for Emacs. */
#define _IO_PENDING_OUTPUT_COUNT(_fp)	\
	((_fp)->_IO_write_ptr - (_fp)->_IO_write_base)

extern void _IO_flockfile (FILE *) __THROW;
extern void _IO_funlockfile (FILE *) __THROW;
extern int _IO_ftrylockfile (FILE *) __THROW;

#define _IO_peekc(_fp) _IO_peekc_unlocked (_fp)
#define _IO_flockfile(_fp) /**/
#define _IO_funlockfile(_fp) ((void) 0)
#define _IO_ftrylockfile(_fp) /**/
#ifndef _IO_cleanup_region_start
#define _IO_cleanup_region_start(_fct, _fp) /**/
#endif
#ifndef _IO_cleanup_region_end
#define _IO_cleanup_region_end(_Doit) /**/
#endif

#define _IO_need_lock(_fp) \
  (((_fp)->_flags2 & _IO_FLAGS2_NEED_LOCK) != 0)

extern int _IO_vfscanf (FILE * __restrict, const char * __restrict,
			__gnuc_va_list, int *__restrict);
extern __ssize_t _IO_padn (FILE *, int, __ssize_t);
extern size_t _IO_sgetn (FILE *, void *, size_t);

extern off64_t _IO_seekoff (FILE *, off64_t, int, int);
extern off64_t _IO_seekpos (FILE *, off64_t, int);

extern void _IO_free_backup_area (FILE *) __THROW;


extern wint_t _IO_getwc (FILE *__fp);
extern wint_t _IO_putwc (wchar_t __wc, FILE *__fp);
extern int _IO_fwide (FILE *__fp, int __mode) __THROW;

#if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_1)
#  define _IO_fwide_maybe_incompatible \
  (__glibc_unlikely (&_IO_stdin_used == NULL))
extern const int _IO_stdin_used;
weak_extern (_IO_stdin_used);
#else
# define _IO_fwide_maybe_incompatible (0)
#endif

/* A special optimized version of the function above.  It optimizes the
   case of initializing an unoriented byte stream.  */
#define _IO_fwide(__fp, __mode) \
  ({ int __result = (__mode);						      \
     if (__result < 0 && ! _IO_fwide_maybe_incompatible)		      \
       {								      \
	 if ((__fp)->_mode == 0)					      \
	   /* We know that all we have to do is to set the flag.  */	      \
	   (__fp)->_mode = -1;						      \
	 __result = (__fp)->_mode;					      \
       }								      \
     else if (__builtin_constant_p (__mode) && (__mode) == 0)		      \
       __result = _IO_fwide_maybe_incompatible ? -1 : (__fp)->_mode;	      \
     else								      \
       __result = _IO_fwide (__fp, __result);				      \
     __result; })

extern __ssize_t _IO_wpadn (FILE *, wint_t, __ssize_t);
extern void _IO_free_wbackup_area (FILE *) __THROW;

#ifdef __LDBL_COMPAT
__LDBL_REDIR_DECL (_IO_vfscanf)
#endif

libc_hidden_proto (__overflow)
libc_hidden_proto (__underflow)
libc_hidden_proto (__uflow)
libc_hidden_proto (__woverflow)
libc_hidden_proto (__wunderflow)
libc_hidden_proto (__wuflow)
libc_hidden_proto (_IO_free_backup_area)
libc_hidden_proto (_IO_free_wbackup_area)
libc_hidden_proto (_IO_padn)
libc_hidden_proto (_IO_putc)
libc_hidden_proto (_IO_sgetn)

#ifdef _IO_MTSAFE_IO
# undef _IO_peekc
# undef _IO_flockfile
# undef _IO_funlockfile
# undef _IO_ftrylockfile

# define _IO_peekc(_fp) _IO_peekc_locked (_fp)
# define _IO_flockfile(_fp) \
  if (((_fp)->_flags & _IO_USER_LOCK) == 0) _IO_lock_lock (*(_fp)->_lock)
# define _IO_funlockfile(_fp) \
  if (((_fp)->_flags & _IO_USER_LOCK) == 0) _IO_lock_unlock (*(_fp)->_lock)
#endif /* _IO_MTSAFE_IO */

#endif /* _LIBIO_H */
