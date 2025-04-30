/* Open a stdio stream on an anonymous, large temporary file.  Generic version.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

#include <fcntl.h>

/* If there is no O_LARGEFILE, then the plain tmpfile definition
   does the job and it gets tmpfile64 as an alias.  */

#if defined O_LARGEFILE && O_LARGEFILE != 0
# define FLAGS		O_LARGEFILE
# define tmpfile	tmpfile64
# include <tmpfile.c>
#endif
