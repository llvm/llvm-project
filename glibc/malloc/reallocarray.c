/* Change the size of an allocated block.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

#include <errno.h>
#include <stdlib.h>
#include <malloc.h>

void *
__libc_reallocarray (void *optr, size_t nmemb, size_t elem_size)
{
  size_t bytes;
  if (__builtin_mul_overflow (nmemb, elem_size, &bytes))
    {
      __set_errno (ENOMEM);
      return 0;
    }
  return realloc (optr, bytes);
}
libc_hidden_def (__libc_reallocarray)

weak_alias (__libc_reallocarray, reallocarray)
