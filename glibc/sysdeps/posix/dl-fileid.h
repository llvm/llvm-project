/* File identity for the dynamic linker.  Generic POSIX.1 version.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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
#include <sys/stat.h>

/* For POSIX.1 systems, the pair of st_dev and st_ino constitute
   a unique identifier for a file.  */
struct r_file_id
  {
    dev_t dev;
    ino64_t ino;
  };

/* Sample FD to fill in *ID.  Returns true on success.
   On error, returns false, with errno set.  */
static inline bool
_dl_get_file_id (int fd, struct r_file_id *id)
{
  struct __stat64_t64 st;

  if (__glibc_unlikely (__fstat64_time64 (fd, &st) < 0))
    return false;

  id->dev = st.st_dev;
  id->ino = st.st_ino;
  return true;
}

/* Compare two results from _dl_get_file_id for equality.  */
static inline bool
_dl_file_id_match_p (const struct r_file_id *a, const struct r_file_id *b)
{
  return a->dev == b->dev && a->ino == b->ino;
}
