/* Initgroups handling in nss_db module.
   Copyright (C) 2011-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gmail.com>.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Library General Public License as
   published by the Free Software Foundation; either version 2 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Library General Public License for more details.

   You should have received a copy of the GNU Library General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

#include <ctype.h>
#include <errno.h>
#include <grp.h>
#include <limits.h>
#include <paths.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/param.h>

#include "nss_db.h"

/* The hashing function we use.  */
#include "../intl/hash-string.h"

enum nss_status
_nss_db_initgroups_dyn (const char *user, gid_t group, long int *start,
			long int *size, gid_t **groupsp, long int limit,
			int *errnop)
{
  struct nss_db_map state = { NULL, 0 };
  enum nss_status status = internal_setent (_PATH_VARDB "group.db", &state);
  if (status != NSS_STATUS_SUCCESS)
    {
      *errnop = errno;
      return status;
    }

  const struct nss_db_header *header = state.header;
  int i;
  for (i = 0; i < header->ndbs; ++i)
    if (header->dbs[i].id == ':')
      break;
  if (i == header->ndbs)
    {
      status = NSS_STATUS_UNAVAIL;
      goto out;
    }

  const stridx_t *hashtable
    = (const stridx_t *) ((const char *) header
			  + header->dbs[i].hashoffset);
  const char *valstrtab = (const char *) header + header->valstroffset;
  size_t userlen = strlen (user);
  uint32_t hashval = __hash_string (user);
  size_t hidx = hashval % header->dbs[i].hashsize;
  size_t hval2 = 1 + hashval % (header->dbs[i].hashsize - 2);

  gid_t *groups = *groupsp;

  status = NSS_STATUS_NOTFOUND;
  while (hashtable[hidx] != ~((stridx_t) 0))
    {
      const char *valstr = valstrtab + hashtable[hidx];
      while (isblank (*valstr))
	++valstr;

      if (strncmp (valstr, user, userlen) == 0 && isblank (valstr[userlen]))
	{
	  valstr += userlen + 1;
	  while (isblank (*valstr))
	    ++valstr;

	  while (*valstr != '\0')
	    {
	      errno = 0;
	      char *endp;
	      unsigned long int n = strtoul (valstr, &endp, 10);
	      if (*endp != ',' && *endp != '\0')
		break;
	      valstr = *endp == '\0' ? endp : endp + 1;

	      if (n != ULONG_MAX || errno != ERANGE)
		{
		  /* Insert the group.  */
		  if (*start == *size)
		    {
		      /* Need a bigger buffer.  */
		      if (limit > 0 && *size == limit)
			{
			  /* We reached the maximum.  */
			  status = NSS_STATUS_SUCCESS;
			  goto out;
			}

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

		  groups[*start] = n;
		  *start += 1;
		}
	    }

	  status = NSS_STATUS_SUCCESS;
	  break;
	}

      if ((hidx += hval2) >= header->dbs[i].hashsize)
	hidx -= header->dbs[i].hashsize;
    }

 out:
  internal_endent (&state);

  return status;
}
