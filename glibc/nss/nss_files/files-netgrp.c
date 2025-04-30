/* Netgroup file parser in nss_files modules.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1996.

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
#include <netdb.h>
#include <stdio.h>
#include <stdio_ext.h>
#include <stdlib.h>
#include <string.h>
#include "nsswitch.h"
#include "netgroup.h"
#include <nss_files.h>

#define DATAFILE	"/etc/netgroup"

libc_hidden_proto (_nss_files_endnetgrent)

#define EXPAND(needed)							      \
  do									      \
    {									      \
      size_t old_cursor = result->cursor - result->data;		      \
      void *old_data = result->data;					      \
									      \
      result->data_size += 512 > 2 * needed ? 512 : 2 * needed;		      \
      result->data = realloc (result->data, result->data_size);		      \
									      \
      if (result->data == NULL)						      \
	{								      \
	  free (old_data);						      \
	  status = NSS_STATUS_UNAVAIL;					      \
	  goto the_end;							      \
	}								      \
									      \
      result->cursor = result->data + old_cursor;			      \
    }									      \
  while (0)


enum nss_status
_nss_files_setnetgrent (const char *group, struct __netgrent *result)
{
  FILE *fp;
  enum nss_status status;

  if (group[0] == '\0')
    return NSS_STATUS_UNAVAIL;

  /* Find the netgroups file and open it.  */
  fp = __nss_files_fopen (DATAFILE);
  if (fp == NULL)
    status = errno == EAGAIN ? NSS_STATUS_TRYAGAIN : NSS_STATUS_UNAVAIL;
  else
    {
      /* Read the file line by line and try to find the description
	 GROUP.  We must take care for long lines.  */
      char *line = NULL;
      size_t line_len = 0;
      const ssize_t group_len = strlen (group);

      status = NSS_STATUS_NOTFOUND;
      result->cursor = result->data;

      while (!__feof_unlocked (fp))
	{
	  ssize_t curlen = __getline (&line, &line_len, fp);
	  int found;

	  if (curlen < 0)
	    {
	      status = NSS_STATUS_NOTFOUND;
	      break;
	    }

	  found = (curlen > group_len && strncmp (line, group, group_len) == 0
		   && isspace (line[group_len]));

	  /* Read the whole line (including continuation) and store it
	     if FOUND in nonzero.  Otherwise we don't need it.  */
	  if (found)
	    {
	      /* Store the data from the first line.  */
	      EXPAND (curlen - group_len);
	      memcpy (result->cursor, &line[group_len + 1],
		      curlen - group_len);
	      result->cursor += (curlen - group_len) - 1;
	    }

	  while (curlen > 1 && line[curlen - 1] == '\n'
		 && line[curlen - 2] == '\\')
	    {
	      /* Yes, we have a continuation line.  */
	      if (found)
		/* Remove these characters from the stored line.  */
		result->cursor -= 2;

	      /* Get next line.  */
	      curlen = __getline (&line, &line_len, fp);
	      if (curlen <= 0)
		break;

	      if (found)
		{
		  /* Make sure we have enough room.  */
		  EXPAND (1 + curlen + 1);

		  /* Add separator in case next line starts immediately.  */
		  *result->cursor++ = ' ';

		  /* Copy new line.  */
		  memcpy (result->cursor, line, curlen + 1);
		  result->cursor += curlen;
		}
	    }

	  if (found)
	    {
	      /* Now we have read the line.  */
	      status = NSS_STATUS_SUCCESS;
	      result->cursor = result->data;
	      result->first = 1;
	      break;
	    }
	}

    the_end:
      /* We don't need the file and the line buffer anymore.  */
      free (line);
      fclose (fp);

      if (status != NSS_STATUS_SUCCESS)
	_nss_files_endnetgrent (result);
    }

  return status;
}
libc_hidden_def (_nss_files_setnetgrent)

enum nss_status
_nss_files_endnetgrent (struct __netgrent *result)
{
  /* Free allocated memory for data if some is present.  */
  free (result->data);
  result->data = NULL;
  result->data_size = 0;
  result->cursor = NULL;
  return NSS_STATUS_SUCCESS;
}
libc_hidden_def (_nss_files_endnetgrent)

static char *
strip_whitespace (char *str)
{
  char *cp = str;

  /* Skip leading spaces.  */
  while (isspace (*cp))
    cp++;

  str = cp;
  while (*cp != '\0' && ! isspace(*cp))
    cp++;

  /* Null-terminate, stripping off any trailing spaces.  */
  *cp = '\0';

  return *str == '\0' ? NULL : str;
}

enum nss_status
_nss_netgroup_parseline (char **cursor, struct __netgrent *result,
			 char *buffer, size_t buflen, int *errnop)
{
  enum nss_status status;
  const char *host, *user, *domain;
  char *cp = *cursor;

  /* Some sanity checks.  */
  if (cp == NULL)
    return NSS_STATUS_NOTFOUND;

  /* First skip leading spaces.  */
  while (isspace (*cp))
    ++cp;

  if (*cp != '(')
    {
      /* We have a list of other netgroups.  */
      char *name = cp;

      while (*cp != '\0' && ! isspace (*cp))
	++cp;

      if (name != cp)
	{
	  /* It is another netgroup name.  */
	  int last = *cp == '\0';

	  result->type = group_val;
	  result->val.group = name;
	  *cp = '\0';
	  if (! last)
	    ++cp;
	  *cursor = cp;
	  result->first = 0;

	  return NSS_STATUS_SUCCESS;
	}

      return result->first ? NSS_STATUS_NOTFOUND : NSS_STATUS_RETURN;
    }

  /* Match host name.  */
  host = ++cp;
  while (*cp != ',')
    if (*cp++ == '\0')
      return result->first ? NSS_STATUS_NOTFOUND : NSS_STATUS_RETURN;

  /* Match user name.  */
  user = ++cp;
  while (*cp != ',')
    if (*cp++ == '\0')
      return result->first ? NSS_STATUS_NOTFOUND : NSS_STATUS_RETURN;

  /* Match domain name.  */
  domain = ++cp;
  while (*cp != ')')
    if (*cp++ == '\0')
      return result->first ? NSS_STATUS_NOTFOUND : NSS_STATUS_RETURN;
  ++cp;


  /* When we got here we have found an entry.  Before we can copy it
     to the private buffer we have to make sure it is big enough.  */
  if (cp - host > buflen)
    {
      *errnop = ERANGE;
      status = NSS_STATUS_TRYAGAIN;
    }
  else
    {
      memcpy (buffer, host, cp - host);
      result->type = triple_val;

      buffer[(user - host) - 1] = '\0';	/* Replace ',' with '\0'.  */
      result->val.triple.host = strip_whitespace (buffer);

      buffer[(domain - host) - 1] = '\0'; /* Replace ',' with '\0'.  */
      result->val.triple.user = strip_whitespace (buffer + (user - host));

      buffer[(cp - host) - 1] = '\0'; /* Replace ')' with '\0'.  */
      result->val.triple.domain = strip_whitespace (buffer + (domain - host));

      status = NSS_STATUS_SUCCESS;

      /* Remember where we stopped reading.  */
      *cursor = cp;

      result->first = 0;
    }

  return status;
}
libc_hidden_def (_nss_netgroup_parseline)


enum nss_status
_nss_files_getnetgrent_r (struct __netgrent *result, char *buffer,
			  size_t buflen, int *errnop)
{
  enum nss_status status;

  status = _nss_netgroup_parseline (&result->cursor, result, buffer, buflen,
				    errnop);

  return status;
}
libc_hidden_def (_nss_files_getnetgrent_r)
