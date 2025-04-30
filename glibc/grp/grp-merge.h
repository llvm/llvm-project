/* Group merging implementation.
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

#ifndef _GRP_MERGE_H
#define _GRP_MERGE_H 1

#include <grp.h>

/* Duplicate a grp struct (and its members). When no longer needed, the
   calling function must free(newbuf).  */
int
__copy_grp (const struct group srcgrp, const size_t buflen,
	    struct group *destgrp, char *destbuf, char **endptr);

/* Merge the member lists of two grp structs together.  */
int
__merge_grp (struct group *savedgrp, char *savedbuf, char *savedend,
	     size_t buflen, struct group *mergegrp, char *mergebuf);

#endif /* _GRP_MERGE_H */
