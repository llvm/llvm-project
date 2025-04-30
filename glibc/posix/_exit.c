/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include <unistd.h>
#include <stdlib.h>

/* The function `_exit' should take a status argument and simply
   terminate program execution, using the low-order 8 bits of the
   given integer as status.  */
void
_exit (int status)
{
  status &= 0xff;
  abort ();
}
libc_hidden_def (_exit)
rtld_hidden_def (_exit)
weak_alias (_exit, _Exit)

stub_warning (_exit)
