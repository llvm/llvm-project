/* Copyright (C) 2004-2021 Free Software Foundation, Inc.
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

#include <stdlib.h>

/* Return 1 if the whole area PTR .. PTR+SIZE is not writable.
   Return -1 if it is writable.  */

int
__readonly_area (const void *ptr, size_t size)
{
  /* We cannot determine in general whether memory is writable or not.
     This must be handled in a system-dependent manner.  to not
     unconditionally break code we need to return here a positive
     answer.  This disables this security measure but that is the
     price people have to pay for using systems without a real
     implementation of this interface.  */
  return 1;
}
