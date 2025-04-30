/* Query filename corresponding to an open FD.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
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

#ifndef _FD_TO_FILENAME_H
#define _FD_TO_FILENAME_H

#include <arch-fd_to_filename.h>
#include <intprops.h>

struct fd_to_filename
{
  /* A positive int value has at most 10 decimal digits.  */
  char buffer[sizeof (FD_TO_FILENAME_PREFIX) + INT_STRLEN_BOUND (int)];
};

/* Writes a /proc/self/fd-style path for DESCRIPTOR to *STORAGE and
   returns a pointer to the start of the string.  DESCRIPTOR must be
   non-negative.  */
char *__fd_to_filename (int descriptor, struct fd_to_filename *storage)
  attribute_hidden;

#endif /* _FD_TO_FILENAME_H */
