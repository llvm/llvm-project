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

#include <ctype.h>
#include <errno.h>
#include <grp.h>
#include <stdio.h>

/* Define a line parsing function using the common code
   used in the nss_files module.  */

#define STRUCTURE	group
#define ENTNAME		grent
struct grent_data {};

#define TRAILING_LIST_MEMBER		gr_mem
#define TRAILING_LIST_SEPARATOR_P(c)	((c) == ',')
#include <nss/nss_files/files-parse.c>
LINE_PARSER
(,
 STRING_FIELD (result->gr_name, ISCOLON, 0);
 if (line[0] == '\0'
     && (result->gr_name[0] == '+' || result->gr_name[0] == '-'))
   {
     result->gr_passwd = NULL;
     result->gr_gid = 0;
   }
 else
   {
     STRING_FIELD (result->gr_passwd, ISCOLON, 0);
     if (result->gr_name[0] == '+' || result->gr_name[0] == '-')
       INT_FIELD_MAYBE_NULL (result->gr_gid, ISCOLON, 0, 10, , 0)
     else
       INT_FIELD (result->gr_gid, ISCOLON, 0, 10,)
   }
 )


/* Read one entry from the given stream.  */
int
__fgetgrent_r (FILE *stream, struct group *resbuf, char *buffer, size_t buflen,
	       struct group **result)
{
  int ret = __nss_fgetent_r (stream, resbuf, buffer, buflen, parse_line);
  if (ret == 0)
    *result = resbuf;
  else
    *result = NULL;
  return ret;
}
weak_alias (__fgetgrent_r, fgetgrent_r)
