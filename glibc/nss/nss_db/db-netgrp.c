/* Netgroup file parser in nss_db modules.
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
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <netgroup.h>
#include <string.h>
#include <stdint.h>
#include <libc-lock.h>
#include <paths.h>
#include <stdlib.h>

#include "nsswitch.h"
#include "nss_db.h"

/* The hashing function we use.  */
#include "../intl/hash-string.h"


#define DBFILE		_PATH_VARDB "netgroup.db"

/* Maintenance of the shared handle open on the database.  */
enum nss_status
_nss_db_setnetgrent (const char *group, struct __netgrent *result)
{
  struct nss_db_map state;
  enum nss_status status = internal_setent (DBFILE, &state);

  if (status == NSS_STATUS_SUCCESS)
    {
      const struct nss_db_header *header = state.header;
      const stridx_t *hashtable
	= (const stridx_t *) ((const char *) header
			      + header->dbs[0].hashoffset);
      const char *valstrtab = (const char *) header + header->valstroffset;
      uint32_t hashval = __hash_string (group);
      size_t grouplen = strlen (group);
      size_t hidx = hashval % header->dbs[0].hashsize;
      size_t hval2 = 1 + hashval % (header->dbs[0].hashsize - 2);

      status = NSS_STATUS_NOTFOUND;
      while (hashtable[hidx] != ~((stridx_t) 0))
	{
	  const char *valstr = valstrtab + hashtable[hidx];

	  if (strncmp (valstr, group, grouplen) == 0
	      && isblank (valstr[grouplen]))
	    {
	      const char *cp = &valstr[grouplen + 1];
	      while (isblank (*cp))
		++cp;
	      if (*cp != '\0')
		{
		  result->data = strdup (cp);
		  if (result->data == NULL)
		    status = NSS_STATUS_TRYAGAIN;
		  else
		    {
		      status = NSS_STATUS_SUCCESS;
		      result->cursor = result->data;
		    }
		  break;
		}
	    }

	  if ((hidx += hval2) >= header->dbs[0].hashsize)
	    hidx -= header->dbs[0].hashsize;
	}

      internal_endent (&state);
    }

  return status;

}


enum nss_status
_nss_db_endnetgrent (struct __netgrent *result)
{
  free (result->data);
  result->data = NULL;
  result->data_size = 0;
  result->cursor = NULL;
  return NSS_STATUS_SUCCESS;
}


extern enum nss_status _nss_netgroup_parseline (char **cursor,
						struct __netgrent *result,
						char *buffer, size_t buflen,
						int *errnop);

enum nss_status
_nss_db_getnetgrent_r (struct __netgrent *result, char *buffer, size_t buflen,
		       int *errnop)
{
  enum nss_status status;

  status = _nss_netgroup_parseline (&result->cursor, result, buffer, buflen,
				    errnop);

  return status;
}
