/* elision-unlock.c: Commit an elided pthread lock.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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

#include "pthreadP.h"
#include "lowlevellock.h"
#include "hle.h"

int
__lll_unlock_elision(int *lock, int private)
{
  /* When the lock was free we're in a transaction.
     When you crash here you unlocked a free lock.  */
  if (*lock == 0)
    _xend();
  else
    lll_unlock ((*lock), private);
  return 0;
}
libc_hidden_def (__lll_unlock_elision)
