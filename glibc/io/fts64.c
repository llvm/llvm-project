/* File tree traversal functions LFS version.
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

#define FTS_OPEN fts64_open
#define FTS_CLOSE fts64_close
#define FTS_READ fts64_read
#define FTS_SET fts64_set
#define FTS_CHILDREN fts64_children
#define FTSOBJ FTS64
#define FTSENTRY FTSENT64
#define INO_T ino64_t
#define STRUCT_STAT stat64
#define STAT __stat64
#define LSTAT __lstat64

#include "fts.c"
