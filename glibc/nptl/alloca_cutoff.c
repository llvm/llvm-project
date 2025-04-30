/* Determine whether block of given size can be allocated on the stack or not.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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

#include <alloca.h>
#include <stdlib.h>
#include <sys/param.h>
#include <pthreadP.h>


int
__libc_alloca_cutoff (size_t size)
{
  return size <= (MIN (__MAX_ALLOCA_CUTOFF,
		       THREAD_GETMEM (THREAD_SELF, stackblock_size) / 4
		       /* The main thread, before the thread library is
			  initialized, has zero in the stackblock_size
			  element.  Since it is the main thread we can
			  assume the maximum available stack space.  */
		       ?: __MAX_ALLOCA_CUTOFF * 4));
}
libc_hidden_def (__libc_alloca_cutoff)
