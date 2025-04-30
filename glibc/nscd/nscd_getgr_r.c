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

#include <alloca.h>
#include <assert.h>
#include <errno.h>
#include <grp.h>
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
#include <scratch_buffer.h>

#include "nscd-client.h"
#include "nscd_proto.h"

int __nss_not_use_nscd_group;

static int nscd_getgr_r (const char *key, size_t keylen, request_type type,
			 struct group *resultbuf, char *buffer,
			 size_t buflen, struct group **result);


int
__nscd_getgrnam_r (const char *name, struct group *resultbuf, char *buffer,
		   size_t buflen, struct group **result)
{
  return nscd_getgr_r (name, strlen (name) + 1, GETGRBYNAME, resultbuf,
		       buffer, buflen, result);
}


int
__nscd_getgrgid_r (gid_t gid, struct group *resultbuf, char *buffer,
		   size_t buflen, struct group **result)
{
  char buf[3 * sizeof (gid_t)];
  buf[sizeof (buf) - 1] = '\0';
  char *cp = _itoa_word (gid, buf + sizeof (buf) - 1, 10, 0);

  return nscd_getgr_r (cp, buf + sizeof (buf) - cp, GETGRBYGID, resultbuf,
		       buffer, buflen, result);
}


libc_locked_map_ptr (,__gr_map_handle) attribute_hidden;
/* Note that we only free the structure if necessary.  The memory
   mapping is not removed since it is not visible to the malloc
   handling.  */
libc_freeres_fn (gr_map_free)
{
  if (__gr_map_handle.mapped != NO_MAPPING)
    {
      void *p = __gr_map_handle.mapped;
      __gr_map_handle.mapped = NO_MAPPING;
      free (p);
    }
}


static int
nscd_getgr_r (const char *key, size_t keylen, request_type type,
	      struct group *resultbuf, char *buffer, size_t buflen,
	      struct group **result)
{
  int gc_cycle;
  int nretries = 0;
  const uint32_t *len = NULL;
  struct scratch_buffer lenbuf;
  scratch_buffer_init (&lenbuf);

  /* If the mapping is available, try to search there instead of
     communicating with the nscd.  */
  struct mapped_database *mapped = __nscd_get_map_ref (GETFDGR, "group",
						       &__gr_map_handle,
						       &gc_cycle);
 retry:;
  const char *gr_name = NULL;
  size_t gr_name_len = 0;
  int retval = -1;
  const char *recend = (const char *) ~UINTMAX_C (0);
  gr_response_header gr_resp;

  if (mapped != NO_MAPPING)
    {
      struct datahead *found = __nscd_cache_search (type, key, keylen, mapped,
						    sizeof gr_resp);
      if (found != NULL)
	{
	  len = (const uint32_t *) (&found->data[0].grdata + 1);
	  gr_resp = found->data[0].grdata;
	  gr_name = ((const char *) len
		     + gr_resp.gr_mem_cnt * sizeof (uint32_t));
	  gr_name_len = gr_resp.gr_name_len + gr_resp.gr_passwd_len;
	  recend = (const char *) found->data + found->recsize;
	  /* Now check if we can trust gr_resp fields.  If GC is
	     in progress, it can contain anything.  */
	  if (mapped->head->gc_cycle != gc_cycle)
	    {
	      retval = -2;
	      goto out;
	    }

	  /* The alignment is always sufficient, unless GC is in progress.  */
	  assert (((uintptr_t) len & (__alignof__ (*len) - 1)) == 0);
	}
    }

  int sock = -1;
  if (gr_name == NULL)
    {
      sock = __nscd_open_socket (key, keylen, type, &gr_resp,
				 sizeof (gr_resp));
      if (sock == -1)
	{
	  __nss_not_use_nscd_group = 1;
	  goto out;
	}
    }

  /* No value found so far.  */
  *result = NULL;

  if (__glibc_unlikely (gr_resp.found == -1))
    {
      /* The daemon does not cache this database.  */
      __nss_not_use_nscd_group = 1;
      goto out_close;
    }

  if (gr_resp.found == 1)
    {
      struct iovec vec[2];
      char *p = buffer;
      size_t total_len;
      uintptr_t align;
      nscd_ssize_t cnt;

      /* Now allocate the buffer the array for the group members.  We must
	 align the pointer.  */
      align = ((__alignof__ (char *) - (p - ((char *) 0)))
	       & (__alignof__ (char *) - 1));
      total_len = (align + (1 + gr_resp.gr_mem_cnt) * sizeof (char *)
		   + gr_resp.gr_name_len + gr_resp.gr_passwd_len);
      if (__glibc_unlikely (buflen < total_len))
	{
	no_room:
	  __set_errno (ERANGE);
	  retval = ERANGE;
	  goto out_close;
	}
      buflen -= total_len;

      p += align;
      resultbuf->gr_mem = (char **) p;
      p += (1 + gr_resp.gr_mem_cnt) * sizeof (char *);

      /* Set pointers for strings.  */
      resultbuf->gr_name = p;
      p += gr_resp.gr_name_len;
      resultbuf->gr_passwd = p;
      p += gr_resp.gr_passwd_len;

      /* Fill in what we know now.  */
      resultbuf->gr_gid = gr_resp.gr_gid;

      /* Read the length information, group name, and password.  */
      if (gr_name == NULL)
	{
	  /* Handle a simple, usual case: no group members.  */
	  if (__glibc_likely (gr_resp.gr_mem_cnt == 0))
	    {
	      size_t n = gr_resp.gr_name_len + gr_resp.gr_passwd_len;
	      if (__builtin_expect (__readall (sock, resultbuf->gr_name, n)
				    != (ssize_t) n, 0))
		goto out_close;
	    }
	  else
	    {
	      /* Allocate array to store lengths.  */
	      if (!scratch_buffer_set_array_size
		  (&lenbuf, gr_resp.gr_mem_cnt, sizeof (uint32_t)))
		goto out_close;
	      len = lenbuf.data;

	      vec[0].iov_base = (void *) len;
	      vec[0].iov_len = gr_resp.gr_mem_cnt * sizeof (uint32_t);
	      vec[1].iov_base = resultbuf->gr_name;
	      vec[1].iov_len = gr_resp.gr_name_len + gr_resp.gr_passwd_len;
	      total_len = vec[0].iov_len + vec[1].iov_len;

	      /* Get this data.  */
	      size_t n = __readvall (sock, vec, 2);
	      if (__glibc_unlikely (n != total_len))
		goto out_close;
	    }
	}
      else
	/* We already have the data.  Just copy the group name and
	   password.  */
	memcpy (resultbuf->gr_name, gr_name,
		gr_resp.gr_name_len + gr_resp.gr_passwd_len);

      /* Clear the terminating entry.  */
      resultbuf->gr_mem[gr_resp.gr_mem_cnt] = NULL;

      /* Prepare reading the group members.  */
      total_len = 0;
      for (cnt = 0; cnt < gr_resp.gr_mem_cnt; ++cnt)
	{
	  resultbuf->gr_mem[cnt] = p;
	  total_len += len[cnt];
	  p += len[cnt];
	}

      if (__glibc_unlikely (gr_name + gr_name_len + total_len > recend))
	{
	  /* len array might contain garbage during nscd GC cycle,
	     retry rather than fail in that case.  */
	  if (gr_name != NULL && mapped->head->gc_cycle != gc_cycle)
	    retval = -2;
	  goto out_close;
	}
      if (__glibc_unlikely (total_len > buflen))
	{
	  /* len array might contain garbage during nscd GC cycle,
	     retry rather than fail in that case.  */
	  if (gr_name != NULL && mapped->head->gc_cycle != gc_cycle)
	    {
	      retval = -2;
	      goto out_close;
	    }
	  else
	    goto no_room;
	}

      retval = 0;

      /* If there are no group members TOTAL_LEN is zero.  */
      if (gr_name == NULL)
	{
	  if (total_len > 0
	      && __builtin_expect (__readall (sock, resultbuf->gr_mem[0],
					      total_len) != total_len, 0))
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
	  /* Copy the group member names.  */
	  memcpy (resultbuf->gr_mem[0], gr_name + gr_name_len, total_len);

	  /* Try to detect corrupt databases.  */
	  if (resultbuf->gr_name[gr_name_len - 1] != '\0'
	      || resultbuf->gr_passwd[gr_resp.gr_passwd_len - 1] != '\0'
	      || ({for (cnt = 0; cnt < gr_resp.gr_mem_cnt; ++cnt)
		    if (resultbuf->gr_mem[cnt][len[cnt] - 1] != '\0')
		      break;
		  cnt < gr_resp.gr_mem_cnt; }))
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

  scratch_buffer_free (&lenbuf);

  return retval;
}
