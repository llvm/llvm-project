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

/* We need to disable the redirect for __fseeko64 for the alias
   definitions below to work.  */
#define __fseeko64 __fseeko64_disable

#include "libioP.h"
#include "stdio.h"

int
__fseeko (FILE *fp, off_t offset, int whence)
{
  int result;
  CHECK_FILE (fp, -1);
  _IO_acquire_lock (fp);
  result = _IO_fseek (fp, offset, whence);
  _IO_release_lock (fp);
  return result;
}
weak_alias (__fseeko, fseeko)

#ifdef __OFF_T_MATCHES_OFF64_T
weak_alias (__fseeko, fseeko64)
# undef __fseeko64
strong_alias (__fseeko, __fseeko64)
libc_hidden_ver (__fseeko, __fseeko64)
#endif
