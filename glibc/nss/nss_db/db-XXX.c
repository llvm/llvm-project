/* Common code for DB-based databases in nss_db module.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
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

#include <dlfcn.h>
#include <fcntl.h>
#include <stdint.h>
#include <sys/mman.h>
#include <libc-lock.h>
#include "nsswitch.h"
#include "nss_db.h"

/* The hashing function we use.  */
#include "../intl/hash-string.h"

/* These symbols are defined by the including source file:

   ENTNAME -- database name of the structure and functions (hostent, pwent).
   STRUCTURE -- struct name, define only if not ENTNAME (passwd, group).
   DATABASE -- database file name, ("hosts", "passwd")

   NEED_H_ERRNO - defined iff an arg `int *herrnop' is used.
*/

#define ENTNAME_r	CONCAT(ENTNAME,_r)

#include <paths.h>
#define	DBFILE		_PATH_VARDB DATABASE ".db"

#ifdef NEED_H_ERRNO
# define H_ERRNO_PROTO	, int *herrnop
# define H_ERRNO_ARG	, herrnop
# define H_ERRNO_SET(val) (*herrnop = (val))
#else
# define H_ERRNO_PROTO
# define H_ERRNO_ARG
# define H_ERRNO_SET(val) ((void) 0)
#endif

/* State for this database.  */
static struct nss_db_map state;
/* Lock to protect the state and global variables.  */
__libc_lock_define (static , lock);

/* Maintenance of the shared handle open on the database.  */
static int keep_db;
static const char *entidx;


/* Open the database.  */
enum nss_status
CONCAT(_nss_db_set,ENTNAME) (int stayopen)
{
  enum nss_status status;

  __libc_lock_lock (lock);

  status = internal_setent (DBFILE, &state);

  if (status == NSS_STATUS_SUCCESS)
    {
      /* Remember STAYOPEN flag.  */
      keep_db |= stayopen;

      /* Reset the sequential index.  */
      entidx  = NULL;
    }

  __libc_lock_unlock (lock);

  return status;
}


/* Close it again.  */
enum nss_status
CONCAT(_nss_db_end,ENTNAME) (void)
{
  __libc_lock_lock (lock);

  internal_endent (&state);

  /* Reset STAYOPEN flag.  */
  keep_db = 0;

  __libc_lock_unlock (lock);

  return NSS_STATUS_SUCCESS;
}


/* Macro for defining lookup functions for this DB-based database.

   NAME is the name of the lookup; e.g. `pwnam'.

   DB_CHAR is index indicator for the database.

   KEYPATTERN gives `printf' args to construct a key string;
   e.g. `("%d", id)'.

   KEYSIZE gives the allocation size of a buffer to construct it in;
   e.g. `1 + sizeof (id) * 4'.

   PROTO is the potentially empty list of other parameters.

   BREAK_IF_MATCH is a block of code which compares `struct STRUCTURE *result'
   to the lookup key arguments and does `break;' if they match.  */

#define DB_LOOKUP(name, db_char, keysize, keypattern, break_if_match, proto...)\
enum nss_status								      \
 _nss_db_get##name##_r (proto, struct STRUCTURE *result,		      \
			char *buffer, size_t buflen, int *errnop H_ERRNO_PROTO)\
{									      \
  struct parser_data *data = (void *) buffer;				      \
									      \
  if (buflen < sizeof *data)						      \
    {									      \
      *errnop = ERANGE;							      \
      H_ERRNO_SET (NETDB_INTERNAL);					      \
      return NSS_STATUS_TRYAGAIN;					      \
    }									      \
									      \
  struct nss_db_map state = { NULL, 0 };				      \
  enum nss_status status = internal_setent (DBFILE, &state);		      \
  if (status != NSS_STATUS_SUCCESS)					      \
    {									      \
      *errnop = errno;							      \
      H_ERRNO_SET (NETDB_INTERNAL);					      \
      return status;							      \
    }									      \
									      \
  const struct nss_db_header *header = state.header;			      \
  int i;								      \
  for (i = 0; i < header->ndbs; ++i)					      \
    if (header->dbs[i].id == db_char)					      \
      break;								      \
  if (i == header->ndbs)						      \
    {									      \
      status = NSS_STATUS_UNAVAIL;					      \
      goto out;								      \
    }									      \
									      \
  char *key;								      \
  if (db_char == '.')							      \
    key = (char *) IGNOREPATTERN keypattern;				      \
  else									      \
    {									      \
      const size_t size = (keysize) + 1;				      \
      key = alloca (size);						      \
									      \
      KEYPRINTF keypattern;						      \
    }									      \
									      \
  const stridx_t *hashtable						      \
    = (const stridx_t *) ((const char *) header				      \
			  + header->dbs[i].hashoffset);			      \
  const char *valstrtab = (const char *) header + header->valstroffset;	      \
  uint32_t hashval = __hash_string (key);				      \
  size_t hidx = hashval % header->dbs[i].hashsize;			      \
  size_t hval2 = 1 + hashval % (header->dbs[i].hashsize - 2);		      \
									      \
  status = NSS_STATUS_NOTFOUND;						      \
  while (hashtable[hidx] != ~((stridx_t) 0))				      \
    {									      \
      const char *valstr = valstrtab + hashtable[hidx];			      \
      size_t len = strlen (valstr) + 1;					      \
      if (len > buflen)							      \
	{								      \
	  /* No room to copy the data to.  */				      \
	  *errnop = ERANGE;						      \
	  H_ERRNO_SET (NETDB_INTERNAL);					      \
	  status = NSS_STATUS_TRYAGAIN;					      \
	  break;							      \
	}								      \
									      \
      /* Copy the string to a place where it can be modified.  */	      \
      char *p = memcpy (buffer, valstr, len);				      \
									      \
      int err = parse_line (p, result, data, buflen, errnop EXTRA_ARGS);      \
									      \
      /* Advance before break_if_match, lest it uses continue to skip
	 to the next entry.  */						      \
      if ((hidx += hval2) >= header->dbs[i].hashsize)			      \
	hidx -= header->dbs[i].hashsize;				      \
									      \
      if (err > 0)							      \
	{								      \
	  status = NSS_STATUS_SUCCESS;					      \
	  break_if_match;						      \
	  status = NSS_STATUS_NOTFOUND;					      \
	}								      \
      else if (err == -1)						      \
	{								      \
	  H_ERRNO_SET (NETDB_INTERNAL);					      \
	  status = NSS_STATUS_TRYAGAIN;					      \
	  break;							      \
	}								      \
    }									      \
									      \
  if (status == NSS_STATUS_NOTFOUND)					      \
    H_ERRNO_SET (HOST_NOT_FOUND);					      \
									      \
 out:									      \
  internal_endent (&state);						      \
									      \
  return status;							      \
}

#define KEYPRINTF(pattern, args...) snprintf (key, size, pattern ,##args)
#define IGNOREPATTERN(pattern, arg1, args...) (char *) (uintptr_t) arg1




/* Return the next entry from the database file, doing locking.  */
enum nss_status
CONCAT(_nss_db_get,ENTNAME_r) (struct STRUCTURE *result, char *buffer,
			       size_t buflen, int *errnop H_ERRNO_PROTO)
{
  /* Return next entry in host file.  */
  enum nss_status status;
  struct parser_data *data = (void *) buffer;

  if (buflen < sizeof *data)
    {
      *errnop = ERANGE;
      H_ERRNO_SET (NETDB_INTERNAL);
      return NSS_STATUS_TRYAGAIN;
    }

  __libc_lock_lock (lock);

  if (state.header == NULL)
    {
      status = internal_setent (DBFILE, &state);
      if (status != NSS_STATUS_SUCCESS)
	{
	  *errnop = errno;
	  H_ERRNO_SET (NETDB_INTERNAL);
	  goto out;
	}
      entidx = NULL;
    }

  /* Start from the beginning if freshly initialized or reset
     requested by set*ent.  */
  if (entidx == NULL)
    entidx = (const char *) state.header + state.header->valstroffset;

  status = NSS_STATUS_UNAVAIL;
  if (state.header != MAP_FAILED)
    {
      const char *const end = ((const char *) state.header
			       + state.header->valstroffset
			       + state.header->valstrlen);
      while (entidx < end)
	{
	  const char *next = rawmemchr (entidx, '\0') + 1;
	  size_t len = next - entidx;

	  if (len > buflen)
	    {
	      /* No room to copy the data to.  */
	      *errnop = ERANGE;
	      H_ERRNO_SET (NETDB_INTERNAL);
	      status = NSS_STATUS_TRYAGAIN;
	      break;
	    }

	  /* Copy the string to a place where it can be modified.  */
	  char *p = memcpy (buffer, entidx, len);

	  int err = parse_line (p, result, data, buflen, errnop EXTRA_ARGS);

	  if (err > 0)
	    {
	      status = NSS_STATUS_SUCCESS;
	      entidx = next;
	      break;
	    }
	  if (err < 0)
	    {
	      H_ERRNO_SET (NETDB_INTERNAL);
	      status = NSS_STATUS_TRYAGAIN;
	      break;
	    }

	  /* Continue with the next record, this one is ill-formed.  */
	  entidx = next;
	}
    }

 out:
  __libc_lock_unlock (lock);

  return status;
}
