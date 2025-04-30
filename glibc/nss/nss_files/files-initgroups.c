/* Initgroups handling in nss_files module.
   Copyright (C) 2011-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <grp.h>
#include <nss.h>
#include <stdio_ext.h>
#include <string.h>
#include <sys/param.h>
#include <stdbool.h>
#include <stdlib.h>
#include <scratch_buffer.h>
#include <nss.h>
#include <nss_files.h>

enum nss_status
_nss_files_initgroups_dyn (const char *user, gid_t group, long int *start,
			   long int *size, gid_t **groupsp, long int limit,
			   int *errnop)
{
  FILE *stream = __nss_files_fopen ("/etc/group");
  if (stream == NULL)
    {
      *errnop = errno;
      return *errnop == ENOMEM ? NSS_STATUS_TRYAGAIN : NSS_STATUS_UNAVAIL;
    }

  char *line = NULL;
  size_t linelen = 0;
  enum nss_status status = NSS_STATUS_SUCCESS;
  bool any = false;

  struct scratch_buffer tmpbuf;
  scratch_buffer_init (&tmpbuf);

  gid_t *groups = *groupsp;

  /* We have to iterate over the entire file.  */
  while (1)
    {
      fpos_t pos;
      fgetpos (stream, &pos);
      ssize_t n = __getline (&line, &linelen, stream);
      if (n < 0)
	{
	  if (! __feof_unlocked (stream))
	    status = ((*errnop = errno) == ENOMEM
		      ? NSS_STATUS_TRYAGAIN : NSS_STATUS_UNAVAIL);
	  break;
	}

      struct group grp;
      int res = _nss_files_parse_grent (line, &grp,
					tmpbuf.data, tmpbuf.length, errnop);
      if (res == -1)
	{
	  if (!scratch_buffer_grow (&tmpbuf))
	    {
	      *errnop = ENOMEM;
	      status = NSS_STATUS_TRYAGAIN;
	      goto out;
	    }
	  /* Reread current line, the parser has clobbered it.  */
	  fsetpos (stream, &pos);
	  continue;
	}

      if (res > 0 && grp.gr_gid != group)
	for (char **m = grp.gr_mem; *m != NULL; ++m)
	  if (strcmp (*m, user) == 0)
	    {
	      /* Matches user.  Insert this group.  */
	      if (*start == *size)
		{
		  /* Need a bigger buffer.  */
		  if (limit > 0 && *size == limit)
		    /* We reached the maximum.  */
		    goto out;

		  long int newsize;
		  if (limit <= 0)
		    newsize = 2 * *size;
		  else
		    newsize = MIN (limit, 2 * *size);

		  gid_t *newgroups = realloc (groups,
					      newsize * sizeof (*groups));
		  if (newgroups == NULL)
		    {
		      *errnop = ENOMEM;
		      status = NSS_STATUS_TRYAGAIN;
		      goto out;
		    }
		  *groupsp = groups = newgroups;
		  *size = newsize;
		}

	      groups[*start] = grp.gr_gid;
	      *start += 1;
	      any = true;

	      break;
	    }
    }

 out:
  /* Free memory.  */
  scratch_buffer_free (&tmpbuf);
  free (line);

  fclose (stream);

  return status == NSS_STATUS_SUCCESS && !any ? NSS_STATUS_NOTFOUND : status;
}
libc_hidden_def (_nss_files_initgroups_dyn)
