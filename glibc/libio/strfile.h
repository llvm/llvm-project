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

#ifndef STRFILE_H_
#define STRFILE_H_

#include "libioP.h"

typedef void *(*_IO_alloc_type) (size_t);
typedef void (*_IO_free_type) (void*);

struct _IO_str_fields
{
  /* These members are preserved for ABI compatibility.  The glibc
     implementation always calls malloc/free for user buffers if
     _IO_USER_BUF or _IO_FLAGS2_USER_WBUF are not set.  */
  _IO_alloc_type _allocate_buffer_unused;
  _IO_free_type _free_buffer_unused;
};

/* This is needed for the Irix6 N32 ABI, which has a 64 bit off_t type,
   but a 32 bit pointer type.  In this case, we get 4 bytes of padding
   after the vtable pointer.  Putting them in a structure together solves
   this problem.  */

struct _IO_streambuf
{
  FILE _f;
  const struct _IO_jump_t *vtable;
};

typedef struct _IO_strfile_
{
  struct _IO_streambuf _sbf;
  struct _IO_str_fields _s;
} _IO_strfile;

/* frozen: set when the program has requested that the array object not
   be altered, reallocated, or freed. */
#define _IO_STR_FROZEN(FP) ((FP)->_f._flags & _IO_USER_BUF)

typedef struct
{
  _IO_strfile f;
  /* This is used for the characters which do not fit in the buffer
     provided by the user.  */
  char overflow_buf[64];
} _IO_strnfile;

extern const struct _IO_jump_t _IO_strn_jumps attribute_hidden;


typedef struct
{
  _IO_strfile f;
  /* This is used for the characters which do not fit in the buffer
     provided by the user.  */
  wchar_t overflow_buf[64];
} _IO_wstrnfile;

extern const struct _IO_jump_t _IO_wstrn_jumps attribute_hidden;

/* Initialize an _IO_strfile SF to read from narrow string STRING, and
   return the corresponding FILE object.  It is not necessary to fclose
   the FILE when it is no longer needed.  */
static inline FILE *
_IO_strfile_read (_IO_strfile *sf, const char *string)
{
  sf->_sbf._f._lock = NULL;
  _IO_no_init (&sf->_sbf._f, _IO_USER_LOCK, -1, NULL, NULL);
  _IO_JUMPS (&sf->_sbf) = &_IO_str_jumps;
  _IO_str_init_static_internal (sf, (char*)string, 0, NULL);
  return &sf->_sbf._f;
}

/* Initialize an _IO_strfile SF and _IO_wide_data WD to read from wide
   string STRING, and return the corresponding FILE object.  It is not
   necessary to fclose the FILE when it is no longer needed.  */
static inline FILE *
_IO_strfile_readw (_IO_strfile *sf, struct _IO_wide_data *wd,
		   const wchar_t *string)
{
  sf->_sbf._f._lock = NULL;
  _IO_no_init (&sf->_sbf._f, _IO_USER_LOCK, 0, wd, &_IO_wstr_jumps);
  _IO_fwide (&sf->_sbf._f, 1);
  _IO_wstr_init_static (&sf->_sbf._f, (wchar_t *)string, 0, NULL);
  return &sf->_sbf._f;
}

#endif /* strfile.h.  */
