/* Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Thorsten Kukuk <kukuk@uni-paderborn.de>, 1998.

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

#include <assert.h>
#include <errno.h>
#include <pwd.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <sys/un.h>
#include <not-cancel.h>
#include <_itoa.h>

#include "nscd-client.h"
#include "nscd_proto.h"

int __nss_not_use_nscd_passwd;

static int nscd_getpw_r (const char *key, size_t keylen, request_type type,
			 struct passwd *resultbuf, char *buffer,
			 size_t buflen, struct passwd **result);

int
__nscd_getpwnam_r (const char *name, struct passwd *resultbuf, char *buffer,
		   size_t buflen, struct passwd **result)
{
  if (name == NULL)
    return -1;

  return nscd_getpw_r (name, strlen (name) + 1, GETPWBYNAME, resultbuf,
		       buffer, buflen, result);
}

int
__nscd_getpwuid_r (uid_t uid, struct passwd *resultbuf, char *buffer,
		   size_t buflen, struct passwd **result)
{
  char buf[3 * sizeof (uid_t)];
  buf[sizeof (buf) - 1] = '\0';
  char *cp = _itoa_word (uid, buf + sizeof (buf) - 1, 10, 0);

  return nscd_getpw_r (cp, buf + sizeof (buf) - cp, GETPWBYUID, resultbuf,
		       buffer, buflen, result);
}


libc_locked_map_ptr (static, map_handle);
/* Note that we only free the structure if necessary.  The memory
   mapping is not removed since it is not visible to the malloc
   handling.  */
libc_freeres_fn (pw_map_free)
{
  if (map_handle.mapped != NO_MAPPING)
    {
      void *p = map_handle.mapped;
      map_handle.mapped = NO_MAPPING;
      free (p);
    }
}


static int
nscd_getpw_r (const char *key, size_t keylen, request_type type,
	      struct passwd *resultbuf, char *buffer, size_t buflen,
	      struct passwd **result)
{
  int gc_cycle;
  int nretries = 0;

  /* If the mapping is available, try to search there instead of
     communicating with the nscd.  */
  struct mapped_database *mapped;
  mapped = __nscd_get_map_ref (GETFDPW, "passwd", &map_handle, &gc_cycle);

 retry:;
  const char *pw_name = NULL;
  int retval = -1;
  const char *recend = (const char *) ~UINTMAX_C (0);
  pw_response_header pw_resp;

  if (mapped != NO_MAPPING)
    {
      struct datahead *found = __nscd_cache_search (type, key, keylen, mapped,
						    sizeof pw_resp);
      if (found != NULL)
	{
	  pw_name = (const char *) (&found->data[0].pwdata + 1);
	  pw_resp = found->data[0].pwdata;
	  recend = (const char *) found->data + found->recsize;
	  /* Now check if we can trust pw_resp fields.  If GC is
	     in progress, it can contain anything.  */
	  if (mapped->head->gc_cycle != gc_cycle)
	    {
	      retval = -2;
	      goto out;
	    }
	}
    }

  int sock = -1;
  if (pw_name == NULL)
    {
      sock = __nscd_open_socket (key, keylen, type, &pw_resp,
				 sizeof (pw_resp));
      if (sock == -1)
	{
	  __nss_not_use_nscd_passwd = 1;
	  goto out;
	}
    }

  /* No value found so far.  */
  *result = NULL;

  if (__glibc_unlikely (pw_resp.found == -1))
    {
      /* The daemon does not cache this database.  */
      __nss_not_use_nscd_passwd = 1;
      goto out_close;
    }

  if (pw_resp.found == 1)
    {
      /* Set the information we already have.  */
      resultbuf->pw_uid = pw_resp.pw_uid;
      resultbuf->pw_gid = pw_resp.pw_gid;

      char *p = buffer;
      /* get pw_name */
      resultbuf->pw_name = p;
      p += pw_resp.pw_name_len;
      /* get pw_passwd */
      resultbuf->pw_passwd = p;
      p += pw_resp.pw_passwd_len;
      /* get pw_gecos */
      resultbuf->pw_gecos = p;
      p += pw_resp.pw_gecos_len;
      /* get pw_dir */
      resultbuf->pw_dir = p;
      p += pw_resp.pw_dir_len;
      /* get pw_pshell */
      resultbuf->pw_shell = p;
      p += pw_resp.pw_shell_len;

      ssize_t total = p - buffer;
      if (__glibc_unlikely (pw_name + total > recend))
	goto out_close;
      if (__glibc_unlikely (buflen < total))
	{
	  __set_errno (ERANGE);
	  retval = ERANGE;
	  goto out_close;
	}

      retval = 0;
      if (pw_name == NULL)
	{
	  ssize_t nbytes = __readall (sock, buffer, total);

	  if (__glibc_unlikely (nbytes != total))
	    {
	      /* The `errno' to some value != ERANGE.  */
	      __set_errno (ENOENT);
	      retval = ENOENT;
	    }
	  else
	    *result = resultbuf;
	}
      else
	{
	  /* Copy the various strings.  */
	  memcpy (resultbuf->pw_name, pw_name, total);

	  /* Try to detect corrupt databases.  */
	  if (resultbuf->pw_name[pw_resp.pw_name_len - 1] != '\0'
	      || resultbuf->pw_passwd[pw_resp.pw_passwd_len - 1] != '\0'
	      || resultbuf->pw_gecos[pw_resp.pw_gecos_len - 1] != '\0'
	      || resultbuf->pw_dir[pw_resp.pw_dir_len - 1] != '\0'
	      || resultbuf->pw_shell[pw_resp.pw_shell_len - 1] != '\0')
	    {
	      /* We cannot use the database.  */
	      retval = mapped->head->gc_cycle != gc_cycle ? -2 : -1;
	      goto out_close;
	    }

	  *result = resultbuf;
	}
    }
  else
    {
      /* Set errno to 0 to indicate no error, just no found record.  */
      __set_errno (0);
      /* Even though we have not found anything, the result is zero.  */
      retval = 0;
    }

 out_close:
  if (sock != -1)
    __close_nocancel_nostatus (sock);
 out:
  if (__nscd_drop_map_ref (mapped, &gc_cycle) != 0)
    {
      /* When we come here this means there has been a GC cycle while we
	 were looking for the data.  This means the data might have been
	 inconsistent.  Retry if possible.  */
      if ((gc_cycle & 1) != 0 || ++nretries == 5 || retval == -1)
	{
	  /* nscd is just running gc now.  Disable using the mapping.  */
	  if (atomic_decrement_val (&mapped->counter) == 0)
	    __nscd_unmap (mapped);
	  mapped = NO_MAPPING;
	}

      if (retval != -1)
	goto retry;
    }

  return retval;
}
