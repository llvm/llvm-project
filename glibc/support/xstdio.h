/* Error-checking wrappers for stdio functions.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#ifndef SUPPORT_XSTDIO_H
#define SUPPORT_XSTDIO_H

#include <stdio.h>
#include <sys/cdefs.h>

__BEGIN_DECLS

FILE *xfopen (const char *path, const char *mode);
void xfclose (FILE *);

/* Read a line from FP, using getline.  *BUFFER must be NULL, or a
   heap-allocated pointer of *LENGTH bytes.  Return the number of
   bytes in the line if a line was read, or 0 on EOF.  */
size_t xgetline (char **lineptr, size_t *n, FILE *stream);

__END_DECLS

#endif /* SUPPORT_XSTDIO_H */
