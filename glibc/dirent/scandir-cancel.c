/* Cancellation handler used in scandir* implementations.
   Copyright (C) 1992-2021 Free Software Foundation, Inc.
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

#include <dirent.h>

void
__scandir_cancel_handler (void *arg)
{
  struct scandir_cancel_struct *cp = arg;
  void **v = cp->v;

  for (size_t i = 0; i < cp->cnt; ++i)
    free (v[i]);
  free (v);
  (void) __closedir (cp->dp);
}
