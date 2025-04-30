/* lockfile - Handle locking and unlocking of streams.  Hurd pthread version.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library;  if not, see
   <https://www.gnu.org/licenses/>.  */

#include <pthread.h>		/* Must come before <stdio.h>! */
#include <stdio.h>

void
_cthreads_flockfile (FILE *fp)
{
  _IO_lock_lock (*fp->_lock);
}

void
_cthreads_funlockfile (FILE *fp)
{
  _IO_lock_unlock (*fp->_lock);
}

int
_cthreads_ftrylockfile (FILE *fp)
{
  return __libc_lock_trylock_recursive (*fp->_lock);
}

#undef	_IO_flockfile
#undef	_IO_funlockfile
#undef	_IO_ftrylockfile
#undef	flockfile
#undef	funlockfile
#undef	ftrylockfile

void _IO_flockfile (FILE *)
     __attribute__ ((alias ("_cthreads_flockfile")));
void _IO_funlockfile (FILE *)
     __attribute__ ((alias ("_cthreads_funlockfile")));
int _IO_ftrylockfile (FILE *)
     __attribute__ ((alias ("_cthreads_ftrylockfile")));

void flockfile (FILE *)
     __attribute__ ((weak, alias ("_cthreads_flockfile")));
void funlockfile (FILE *)
     __attribute__ ((weak, alias ("_cthreads_funlockfile")));
int ftrylockfile (FILE *)
     __attribute__ ((weak, alias ("_cthreads_ftrylockfile")));
