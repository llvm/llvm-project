/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
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
#include <shadow.h>
#include <stdio.h>
#include <string.h>

/* Define a line parsing function using the common code
   used in the nss_files module.  */

#define STRUCTURE	spwd
#define ENTNAME		spent
struct spent_data {};

/* Predicate which always returns false, needed below.  */
#define FALSEP(arg) 0


#include <nss/nss_files/files-parse.c>
LINE_PARSER
(,
 STRING_FIELD (result->sp_namp, ISCOLON, 0);
 if (line[0] == '\0'
     && (result->sp_namp[0] == '+' || result->sp_namp[0] == '-'))
   {
     result->sp_pwdp = NULL;
     result->sp_lstchg = 0;
     result->sp_min = 0;
     result->sp_max = 0;
     result->sp_warn = -1l;
     result->sp_inact = -1l;
     result->sp_expire = -1l;
     result->sp_flag = ~0ul;
   }
 else
   {
     STRING_FIELD (result->sp_pwdp, ISCOLON, 0);
     INT_FIELD_MAYBE_NULL (result->sp_lstchg, ISCOLON, 0, 10, (long int) (int),
			   (long int) -1);
     INT_FIELD_MAYBE_NULL (result->sp_min, ISCOLON, 0, 10, (long int) (int),
			   (long int) -1);
     INT_FIELD_MAYBE_NULL (result->sp_max, ISCOLON, 0, 10, (long int) (int),
			   (long int) -1);
     while (isspace (*line))
       ++line;
     if (*line == '\0')
       {
	 /* The old form.  */
	 result->sp_warn = -1l;
	 result->sp_inact = -1l;
	 result->sp_expire = -1l;
	 result->sp_flag = ~0ul;
       }
     else
       {
	 INT_FIELD_MAYBE_NULL (result->sp_warn, ISCOLON, 0, 10,
			       (long int) (int), (long int) -1);
	 INT_FIELD_MAYBE_NULL (result->sp_inact, ISCOLON, 0, 10,
			       (long int) (int), (long int) -1);
	 INT_FIELD_MAYBE_NULL (result->sp_expire, ISCOLON, 0, 10,
			       (long int) (int), (long int) -1);
	 if (*line != '\0')
	   INT_FIELD_MAYBE_NULL (result->sp_flag, FALSEP, 0, 10,
				 (unsigned long int), ~0ul)
	 else
	   result->sp_flag = ~0ul;
       }
   }
 )


/* Read one shadow entry from the given stream.  */
int
__sgetspent_r (const char *string, struct spwd *resbuf, char *buffer,
	       size_t buflen, struct spwd **result)
{
  buffer[buflen - 1] = '\0';
  char *sp = strncpy (buffer, string, buflen);
  if (buffer[buflen - 1] != '\0')
    return ERANGE;

  int parse_result = parse_line (sp, resbuf, NULL, 0, &errno);
  *result = parse_result > 0 ? resbuf : NULL;

  return *result == NULL ? errno : 0;
}
weak_alias (__sgetspent_r, sgetspent_r)
