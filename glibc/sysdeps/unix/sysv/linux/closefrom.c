/* Close a range of file descriptors.  Linux version.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#include <stdbool.h>
#include <stdio.h>
#include <sys/param.h>
#include <unistd.h>

void
__closefrom (int lowfd)
{
  int l = MAX (0, lowfd);

  int r = __close_range (l, ~0U, 0);
  if (r == 0)
    return;

  if (!__closefrom_fallback (l, true))
    __fortify_fail ("closefrom failed to close a file descriptor");
}
weak_alias (__closefrom, closefrom)
