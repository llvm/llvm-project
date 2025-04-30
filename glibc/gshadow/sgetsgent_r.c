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
#include <string.h>

/* Define a line parsing function using the common code
   used in the nss_files module.  */

#define STRUCTURE	sgrp
#define ENTNAME		sgent
struct sgent_data {};


#define TRAILING_LIST_MEMBER		sg_mem
#define TRAILING_LIST_SEPARATOR_P(c)	((c) == ',')
#include <nss/nss_files/files-parse.c>
LINE_PARSER
(,
 STRING_FIELD (result->sg_namp, ISCOLON, 0);
 if (line[0] == '\0'
     && (result->sg_namp[0] == '+' || result->sg_namp[0] == '-'))
   {
     result->sg_passwd = NULL;
     result->sg_adm = NULL;
     result->sg_mem = NULL;
   }
 else
   {
     STRING_FIELD (result->sg_passwd, ISCOLON, 0);
     STRING_LIST (result->sg_adm, ':');
   }
 )


/* Read one shadow entry from the given stream.  */
int
__sgetsgent_r (const char *string, struct sgrp *resbuf, char *buffer,
	       size_t buflen, struct sgrp **result)
{
  char *sp;
  if (string < buffer || string >= buffer + buflen)
    {
      buffer[buflen - 1] = '\0';
      sp = strncpy (buffer, string, buflen);
      if (buffer[buflen - 1] != '\0')
	return ERANGE;
    }
  else
    sp = (char *) string;

  int parse_result = parse_line (sp, resbuf, (void *) buffer, buflen, &errno);
  *result = parse_result > 0 ? resbuf : NULL;

  return *result == NULL ? errno : 0;
}
weak_alias (__sgetsgent_r, sgetsgent_r)
