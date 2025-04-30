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

#include <errno.h>
#include <signal.h>
#include <sigsetops.h>

/* Combine sets LEFT and RIGHT by logical AND and place result in DEST.  */
int
sigandset (sigset_t *dest, const sigset_t *left, const sigset_t *right)
{
  if (!dest || !left || !right)
    {
      __set_errno (EINVAL);
      return -1;
    }

  __sigandset (dest, left, right);
  return 0;
}
