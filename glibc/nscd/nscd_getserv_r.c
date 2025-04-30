/* Copyright (C) 2007-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2007.

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
#include <string.h>
#include <not-cancel.h>
#include <_itoa.h>
#include <stdint.h>

#include "nscd-client.h"
#include "nscd_proto.h"


int __nss_not_use_nscd_services;


static int nscd_getserv_r (const char *crit, size_t critlen, const char *proto,
			   request_type type, struct servent *resultbuf,
			   char *buf, size_t buflen, struct servent **result);


int
__nscd_getservbyname_r (const char *name, const char *proto,
			struct servent *result_buf, char *buf, size_t buflen,
			struct servent **result)
{
  return nscd_getserv_r (name, strlen (name), proto, GETSERVBYNAME, result_buf,
			 buf, buflen, result);
}


int
__nscd_getservbyport_r (int port, const char *proto,
			struct servent *result_buf, char *buf, size_t buflen,
			struct servent **result)
{
  char portstr[3 * sizeof (int) + 2];
  portstr[sizeof (portstr) - 1] = '\0';
  char *cp = _itoa_word (port, portstr + sizeof (portstr) - 1, 10, 0);

  return nscd_getserv_r (cp, portstr + sizeof (portstr) - 1 - cp, proto,
			 GETSERVBYPORT, result_buf, buf, buflen, result);
}


libc_locked_map_ptr (, __serv_map_handle) attribute_hidden;
/* Note that we only free the structure if necessary.  The memory
   mapping is not removed since it is not visible to the malloc
   handling.  */
libc_freeres_fn (serv_map_free)
{
  if (__serv_map_handle.mapped != NO_MAPPING)
    {
      void *p = __serv_map_handle.mapped;
      __serv_map_handle.mapped = NO_MAPPING;
      free (p);
    }
}


static int
nscd_getserv_r (const char *crit, size_t critlen, const char *proto,
		request_type type, struct servent *resultbuf,
		char *buf, size_t buflen, struct servent **result)
{
  int gc_cycle;
  int nretries = 0;
  size_t alloca_used = 0;

  /* If the mapping is available, try to search there instead of
     communicating with the nscd.  */
  struct mapped_database *mapped;
  mapped = __nscd_get_map_ref (GETFDSERV, "services", &__serv_map_handle,
			       &gc_cycle);
  size_t protolen = proto == NULL ? 0 : strlen (proto);
  size_t keylen = critlen + 1 + protolen + 1;
  int alloca_key = __libc_use_alloca (keylen);
  char *key;
  if (alloca_key)
    key = alloca_account (keylen, alloca_used);
  else
    {
      key = malloc (keylen);
      if (key == NULL)
	return -1;
    }
  memcpy (__mempcpy (__mempcpy (key, crit, critlen),
		     "/", 1), proto ?: "", protolen + 1);

 retry:;
  const char *s_name = NULL;
  const char *s_proto = NULL;
  int alloca_aliases_len = 0;
  const uint32_t *aliases_len = NULL;
  const char *aliases_list = NULL;
  int retval = -1;
  const char *recend = (const char *) ~UINTMAX_C (0);
  int sock = -1;
  serv_response_header serv_resp;

  if (mapped != NO_MAPPING)
    {
      struct datahead *found = __nscd_cache_search (type, key, keylen, mapped,
						    sizeof serv_resp);

      if (found != NULL)
	{
	  s_name = (char *) (&found->data[0].servdata + 1);
	  serv_resp = found->data[0].servdata;
	  s_proto = s_name + serv_resp.s_name_len;
	  alloca_aliases_len = 1;
	  aliases_len = (uint32_t *) (s_proto + serv_resp.s_proto_len);
	  aliases_list = ((char *) aliases_len
			  + serv_resp.s_aliases_cnt * sizeof (uint32_t));
	  recend = (const char *) found->data + found->recsize;
	  /* Now check if we can trust serv_resp fields.  If GC is
	     in progress, it can contain anything.  */
	  if (mapped->head->gc_cycle != gc_cycle)
	    {
	      retval = -2;
	      goto out;
	    }
	  if (__builtin_expect ((const char *) aliases_len
				+ serv_resp.s_aliases_cnt * sizeof (uint32_t)
				> recend, 0))
	    goto out;

#if !_STRING_ARCH_unaligned
	  /* The aliases_len array in the mapped database might very
	     well be unaligned.  We will access it word-wise so on
	     platforms which do not tolerate unaligned accesses we
	     need to make an aligned copy.  */
	  if (((uintptr_t) aliases_len & (__alignof__ (*aliases_len) - 1))
	      != 0)
	    {
	      uint32_t *tmp;
	      alloca_aliases_len
		= __libc_use_alloca (alloca_used
				     + (serv_resp.s_aliases_cnt
					* sizeof (uint32_t)));
	      if (alloca_aliases_len)
		tmp = alloca_account (serv_resp.s_aliases_cnt
				      * sizeof (uint32_t),
				      alloca_used);
	      else
		{
		  tmp = malloc (serv_resp.s_aliases_cnt * sizeof (uint32_t));
		  if (tmp == NULL)
		    {
		      retval = ENOMEM;
		      goto out;
		    }
		}
	      aliases_len = memcpy (tmp, aliases_len,
				    serv_resp.s_aliases_cnt
				    * sizeof (uint32_t));
	    }
#endif
	}
    }

  if (s_name == NULL)
    {
      sock = __nscd_open_socket (key, keylen, type, &serv_resp,
				 sizeof (serv_resp));
      if (sock == -1)
	{
	  __nss_not_use_nscd_services = 1;
	  goto out;
	}
    }

  /* No value found so far.  */
  *result = NULL;

  if (__glibc_unlikely (serv_resp.found == -1))
    {
      /* The daemon does not cache this database.  */
      __nss_not_use_nscd_services = 1;
      goto out_close;
    }

  if (serv_resp.found == 1)
    {
      char *cp = buf;
      uintptr_t align1;
      uintptr_t align2;
      size_t total_len;
      ssize_t cnt;
      int n;

      /* A first check whether the buffer is sufficiently large is possible.  */
      /* Now allocate the buffer the array for the group members.  We must
	 align the pointer and the base of the h_addr_list pointers.  */
      align1 = ((__alignof__ (char *) - (cp - ((char *) 0)))
		& (__alignof__ (char *) - 1));
      align2 = ((__alignof__ (char *) - ((cp + align1 + serv_resp.s_name_len
					  + serv_resp.s_proto_len)
					 - ((char *) 0)))
		& (__alignof__ (char *) - 1));
      if (buflen < (align1 + serv_resp.s_name_len + serv_resp.s_proto_len
		    + align2
		    + (serv_resp.s_aliases_cnt + 1) * sizeof (char *)))
	{
	no_room:
	  __set_errno (ERANGE);
	  retval = ERANGE;
	  goto out_close;
	}
      cp += align1;

      /* Prepare the result as far as we can.  */
      resultbuf->s_aliases = (char **) cp;
      cp += (serv_resp.s_aliases_cnt + 1) * sizeof (char *);

      resultbuf->s_name = cp;
      cp += serv_resp.s_name_len;
      resultbuf->s_proto = cp;
      cp += serv_resp.s_proto_len + align2;
      resultbuf->s_port = serv_resp.s_port;

      if (s_name == NULL)
	{
	  struct iovec vec[2];

	  vec[0].iov_base = resultbuf->s_name;
	  vec[0].iov_len = serv_resp.s_name_len + serv_resp.s_proto_len;
	  total_len = vec[0].iov_len;
	  n = 1;

	  if (serv_resp.s_aliases_cnt > 0)
	    {
	      assert (alloca_aliases_len == 0);
	      alloca_aliases_len
		= __libc_use_alloca (alloca_used
				     + (serv_resp.s_aliases_cnt
					* sizeof (uint32_t)));
	      if (alloca_aliases_len)
		aliases_len = alloca_account (serv_resp.s_aliases_cnt
					      * sizeof (uint32_t),
					      alloca_used);
	      else
		{
		  aliases_len = malloc (serv_resp.s_aliases_cnt
					* sizeof (uint32_t));
		  if (aliases_len == NULL)
		    {
		      retval = ENOMEM;
		      goto out_close;
		    }
		}
	      vec[n].iov_base = (void *) aliases_len;
	      vec[n].iov_len = serv_resp.s_aliases_cnt * sizeof (uint32_t);

	      total_len += serv_resp.s_aliases_cnt * sizeof (uint32_t);
	      ++n;
	    }

	  if ((size_t) __readvall (sock, vec, n) != total_len)
	    goto out_close;
	}
      else
	memcpy (resultbuf->s_name, s_name,
		serv_resp.s_name_len + serv_resp.s_proto_len);

      /*  Now we also can read the aliases.  */
      total_len = 0;
      for (cnt = 0; cnt < serv_resp.s_aliases_cnt; ++cnt)
	{
	  resultbuf->s_aliases[cnt] = cp;
	  cp += aliases_len[cnt];
	  total_len += aliases_len[cnt];
	}
      resultbuf->s_aliases[cnt] = NULL;

      if (__builtin_expect ((const char *) aliases_list + total_len > recend,
			    0))
	{
	  /* aliases_len array might contain garbage during nscd GC cycle,
	     retry rather than fail in that case.  */
	  if (aliases_list != NULL && mapped->head->gc_cycle != gc_cycle)
	    retval = -2;
	  goto out_close;
	}

      /* See whether this would exceed the buffer capacity.  */
      if (__glibc_unlikely (cp > buf + buflen))
	{
	  /* aliases_len array might contain garbage during nscd GC cycle,
	     retry rather than fail in that case.  */
	  if (aliases_list != NULL && mapped->head->gc_cycle != gc_cycle)
	    {
	      retval = -2;
	      goto out_close;
	    }
	  goto no_room;
	}

      /* And finally read the aliases.  */
      if (aliases_list == NULL)
	{
	  if (total_len == 0
	      || ((size_t) __readall (sock, resultbuf->s_aliases[0], total_len)
		  == total_len))
	    {
	      retval = 0;
	      *result = resultbuf;
	    }
	}
      else
	{
	  memcpy (resultbuf->s_aliases[0], aliases_list, total_len);

	  /* Try to detect corrupt databases.  */
	  if (resultbuf->s_name[serv_resp.s_name_len - 1] != '\0'
	      || resultbuf->s_proto[serv_resp.s_proto_len - 1] != '\0'
	      || ({for (cnt = 0; cnt < serv_resp.s_aliases_cnt; ++cnt)
		     if (resultbuf->s_aliases[cnt][aliases_len[cnt] - 1]
			 != '\0')
		       break;
		   cnt < serv_resp.s_aliases_cnt; }))
	    {
	      /* We cannot use the database.  */
	      if (mapped->head->gc_cycle != gc_cycle)
		retval = -2;
	      goto out_close;
	    }

	  retval = 0;
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
	{
	  if (!alloca_aliases_len)
	    free ((void *) aliases_len);
	  goto retry;
	}
    }

  if (!alloca_aliases_len)
    free ((void *) aliases_len);
  if (!alloca_key)
    free (key);

  return retval;
}
