/* File identity for the dynamic linker.  Stub version.
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

/* This type stores whatever information is fetched by _dl_get_file_id
   and compared by _dl_file_id_match_p.  */
struct r_file_id
  {
    /* In the stub version, we don't record anything at all.  */
  };

/* Sample FD to fill in *ID.  Returns true on success.
   On error, returns false, with errno set.  */
static inline bool
_dl_get_file_id (int fd __attribute__ ((unused)),
		 struct r_file_id *id __attribute__ ((unused)))
{
  return true;
}

/* Compare two results from _dl_get_file_id for equality.
   It's crucial that this never return false-positive matches.
   It's ideal that it never return false-negative mismatches either,
   but lack of matches is survivable.  */
static inline bool
_dl_file_id_match_p (const struct r_file_id *a __attribute__ ((unused)),
		     const struct r_file_id *b __attribute__ ((unused)))
{
  return false;
}
