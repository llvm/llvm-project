/* Copyright (C) 2009-2021 Free Software Foundation, Inc.
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

#include <ctype.h>
#include <errno.h>
#include <gshadow.h>
#include <stdio.h>

/* Define a line parsing function using the common code
   used in the nss_files module.  */

#define STRUCTURE	sgrp
#define ENTNAME		sgent
#define	EXTERN_PARSER	1
struct sgent_data {};

#include <nss/nss_files/files-parse.c>


/* Read one shadow entry from the given stream.  */
int
__fgetsgent_r (FILE *stream, struct sgrp *resbuf, char *buffer, size_t buflen,
	       struct sgrp **result)
{
  int ret = __nss_fgetent_r (stream, resbuf, buffer, buflen, parse_line);
  if (ret == 0)
    *result = resbuf;
  else
    *result = NULL;
  return ret;
}
weak_alias (__fgetsgent_r, fgetsgent_r)
