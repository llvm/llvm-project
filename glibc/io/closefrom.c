/* Close a range of file descriptors.
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

#include <stdio.h>
#include <unistd.h>
#include <not-cancel.h>

void
__closefrom (int lowfd)
{
  int maxfd = __getdtablesize ();
  if (maxfd == -1)
    __fortify_fail ("closefrom failed to get the file descriptor table size");

  for (int i = 0; i < maxfd; i++)
    if (i >= lowfd)
      __close_nocancel_nostatus (i);
}
weak_alias (__closefrom, closefrom)
