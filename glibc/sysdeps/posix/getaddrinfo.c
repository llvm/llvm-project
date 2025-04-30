/* Host and service name lookups using Name Service Switch modules.
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

/* The Inner Net License, Version 2.00

  The author(s) grant permission for redistribution and use in source and
binary forms, with or without modification, of the software and documentation
provided that the following conditions are met:

0. If you receive a version of the software that is specifically labelled
   as not being for redistribution (check the version message and/or README),
   you are not permitted to redistribute that version of the software in any
   way or form.
1. All terms of the all other applicable copyrights and licenses must be
   followed.
2. Redistributions of source code must retain the authors' copyright
   notice(s), this list of conditions, and the following disclaimer.
3. Redistributions in binary form must reproduce the authors' copyright
   notice(s), this list of conditions, and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
4. [The copyright holder has authorized the removal of this clause.]
5. Neither the name(s) of the author(s) nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY ITS AUTHORS AND CONTRIBUTORS ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  If these license terms cause you a real problem, contact the author.  */

/* This software is Copyright 1996 by Craig Metz, All Rights Reserved.  */

#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <nss.h>
#include <resolv/resolv-internal.h>
#include <resolv/resolv_context.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdio_ext.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/utsname.h>
#include <unistd.h>
#include <nsswitch.h>
#include <libc-diag.h>
#include <libc-lock.h>
#include <not-cancel.h>
#include <nscd/nscd-client.h>
#include <nscd/nscd_proto.h>
#include <scratch_buffer.h>
#include <inet/net-internal.h>

/* Former AI_IDN_ALLOW_UNASSIGNED and AI_IDN_USE_STD3_ASCII_RULES
   flags, now ignored.  */
#define DEPRECATED_AI_IDN 0x300

#if defined(__clang__)
DIAG_IGNORE_NEEDS_COMMENT (4.7, "-Wsometimes-uninitialized");
#else
DIAG_IGNORE_NEEDS_COMMENT (4.7, "-Wmaybe-uninitialized");
#endif

#if IS_IN (libc)
# define feof_unlocked(fp) __feof_unlocked (fp)
#endif

struct gaih_service
  {
    const char *name;
    int num;
  };

struct gaih_servtuple
  {
    struct gaih_servtuple *next;
    int socktype;
    int protocol;
    int port;
  };

static const struct gaih_servtuple nullserv;


struct gaih_typeproto
  {
    int socktype;
    int protocol;
    uint8_t protoflag;
    bool defaultflag;
    char name[8];
  };

/* Values for `protoflag'.  */
#define GAI_PROTO_NOSERVICE	1
#define GAI_PROTO_PROTOANY	2

static const struct gaih_typeproto gaih_inet_typeproto[] =
{
  { 0, 0, 0, false, "" },
  { SOCK_STREAM, IPPROTO_TCP, 0, true, "tcp" },
  { SOCK_DGRAM, IPPROTO_UDP, 0, true, "udp" },
#if defined SOCK_DCCP && defined IPPROTO_DCCP
  { SOCK_DCCP, IPPROTO_DCCP, 0, false, "dccp" },
#endif
#ifdef IPPROTO_UDPLITE
  { SOCK_DGRAM, IPPROTO_UDPLITE, 0, false, "udplite" },
#endif
#ifdef IPPROTO_SCTP
  { SOCK_STREAM, IPPROTO_SCTP, 0, false, "sctp" },
  { SOCK_SEQPACKET, IPPROTO_SCTP, 0, false, "sctp" },
#endif
  { SOCK_RAW, 0, GAI_PROTO_PROTOANY|GAI_PROTO_NOSERVICE, true, "raw" },
  { 0, 0, 0, false, "" }
};

static const struct addrinfo default_hints =
  {
    .ai_flags = AI_DEFAULT,
    .ai_family = PF_UNSPEC,
    .ai_socktype = 0,
    .ai_protocol = 0,
    .ai_addrlen = 0,
    .ai_addr = NULL,
    .ai_canonname = NULL,
    .ai_next = NULL
  };


static int
gaih_inet_serv (const char *servicename, const struct gaih_typeproto *tp,
		const struct addrinfo *req, struct gaih_servtuple *st,
		struct scratch_buffer *tmpbuf)
{
  struct servent *s;
  struct servent ts;
  int r;

  do
    {
      r = __getservbyname_r (servicename, tp->name, &ts,
			     tmpbuf->data, tmpbuf->length, &s);
      if (r != 0 || s == NULL)
	{
	  if (r == ERANGE)
	    {
	      if (!scratch_buffer_grow (tmpbuf))
		return -EAI_MEMORY;
	    }
	  else
	    return -EAI_SERVICE;
	}
    }
  while (r);

  st->next = NULL;
  st->socktype = tp->socktype;
  st->protocol = ((tp->protoflag & GAI_PROTO_PROTOANY)
		  ? req->ai_protocol : tp->protocol);
  st->port = s->s_port;

  return 0;
}

/* Convert struct hostent to a list of struct gaih_addrtuple objects.
   h_name is not copied, and the struct hostent object must not be
   deallocated prematurely.  *RESULT must be NULL or a pointer to a
   linked-list.  The new addresses are appended at the end.  */
static bool
convert_hostent_to_gaih_addrtuple (const struct addrinfo *req,
				   int family,
				   struct hostent *h,
				   struct gaih_addrtuple **result)
{
  while (*result)
    result = &(*result)->next;

  /* Count the number of addresses in h->h_addr_list.  */
  size_t count = 0;
  for (char **p = h->h_addr_list; *p != NULL; ++p)
    ++count;

  /* Report no data if no addresses are available, or if the incoming
     address size is larger than what we can store.  */
  if (count == 0 || h->h_length > sizeof (((struct gaih_addrtuple) {}).addr))
    return true;

  struct gaih_addrtuple *array = calloc (count, sizeof (*array));
  if (array == NULL)
    return false;

  for (size_t i = 0; i < count; ++i)
    {
      if (family == AF_INET && req->ai_family == AF_INET6)
	{
	  /* Perform address mapping. */
	  array[i].family = AF_INET6;
	  memcpy(array[i].addr + 3, h->h_addr_list[i], sizeof (uint32_t));
	  array[i].addr[2] = htonl (0xffff);
	}
      else
	{
	  array[i].family = family;
	  memcpy (array[i].addr, h->h_addr_list[i], h->h_length);
	}
      array[i].next = array + i + 1;
    }
  array[0].name = h->h_name;
  array[count - 1].next = NULL;

  *result = array;
  return true;
}

#define gethosts(_family, _type) \
 {									      \
  struct hostent th;							      \
  char *localcanon = NULL;						      \
  no_data = 0;								      \
  while (1)								      \
    {									      \
      status = DL_CALL_FCT (fct, (name, _family, &th,			      \
				  tmpbuf->data, tmpbuf->length,		      \
				  &errno, &h_errno, NULL, &localcanon));      \
      if (status != NSS_STATUS_TRYAGAIN || h_errno != NETDB_INTERNAL	      \
	  || errno != ERANGE)						      \
	break;								      \
      if (!scratch_buffer_grow (tmpbuf))				      \
	{								      \
	  __resolv_context_put (res_ctx);				      \
	  result = -EAI_MEMORY;						      \
	  goto free_and_return;						      \
	}								      \
    }									      \
  if (status == NSS_STATUS_NOTFOUND					      \
      || status == NSS_STATUS_TRYAGAIN || status == NSS_STATUS_UNAVAIL)	      \
    {									      \
      if (h_errno == NETDB_INTERNAL)					      \
	{								      \
	  __resolv_context_put (res_ctx);				      \
	  result = -EAI_SYSTEM;						      \
	  goto free_and_return;						      \
	}								      \
      if (h_errno == TRY_AGAIN)						      \
	no_data = EAI_AGAIN;						      \
      else								      \
	no_data = h_errno == NO_DATA;					      \
    }									      \
  else if (status == NSS_STATUS_SUCCESS)				      \
    {									      \
      if (!convert_hostent_to_gaih_addrtuple (req, _family, &th, &addrmem))   \
	{								      \
	  __resolv_context_put (res_ctx);				      \
	  result = -EAI_SYSTEM;						      \
	  goto free_and_return;						      \
	}								      \
      *pat = addrmem;							      \
									      \
      if (localcanon != NULL && canon == NULL)				      \
	{								      \
	  canonbuf = __strdup (localcanon);				      \
	  if (canonbuf == NULL)						      \
	    {								      \
	      __resolv_context_put (res_ctx);				      \
	      result = -EAI_SYSTEM;					      \
	      goto free_and_return;					      \
	    }								      \
	  canon = canonbuf;						      \
	}								      \
      if (_family == AF_INET6 && *pat != NULL)				      \
	got_ipv6 = true;						      \
    }									      \
 }


/* This function is called if a canonical name is requested, but if
   the service function did not provide it.  It tries to obtain the
   name using getcanonname_r from the same service NIP.  If the name
   cannot be canonicalized, return a copy of NAME.  Return NULL on
   memory allocation failure.  The returned string is allocated on the
   heap; the caller has to free it.  */
static char *
getcanonname (nss_action_list nip, struct gaih_addrtuple *at, const char *name)
{
  nss_getcanonname_r *cfct = __nss_lookup_function (nip, "getcanonname_r");
  char *s = (char *) name;
  if (cfct != NULL)
    {
      char buf[256];
      if (DL_CALL_FCT (cfct, (at->name ?: name, buf, sizeof (buf),
			      &s, &errno, &h_errno)) != NSS_STATUS_SUCCESS)
	/* If the canonical name cannot be determined, use the passed
	   string.  */
	s = (char *) name;
    }
  return __strdup (name);
}

static int
gaih_inet (const char *name, const struct gaih_service *service,
	   const struct addrinfo *req, struct addrinfo **pai,
	   unsigned int *naddrs, struct scratch_buffer *tmpbuf)
{
  const struct gaih_typeproto *tp = gaih_inet_typeproto;
  struct gaih_servtuple *st = (struct gaih_servtuple *) &nullserv;
  struct gaih_addrtuple *at = NULL;
  bool got_ipv6 = false;
  const char *canon = NULL;
  const char *orig_name = name;

  /* Reserve stack memory for the scratch buffer in the getaddrinfo
     function.  */
  size_t alloca_used = sizeof (struct scratch_buffer);

  if (req->ai_protocol || req->ai_socktype)
    {
      ++tp;

      while (tp->name[0]
	     && ((req->ai_socktype != 0 && req->ai_socktype != tp->socktype)
		 || (req->ai_protocol != 0
		     && !(tp->protoflag & GAI_PROTO_PROTOANY)
		     && req->ai_protocol != tp->protocol)))
	++tp;

      if (! tp->name[0])
	{
	  if (req->ai_socktype)
	    return -EAI_SOCKTYPE;
	  else
	    return -EAI_SERVICE;
	}
    }

  int port = 0;
  if (service != NULL)
    {
      if ((tp->protoflag & GAI_PROTO_NOSERVICE) != 0)
	return -EAI_SERVICE;

      if (service->num < 0)
	{
	  if (tp->name[0])
	    {
	      st = (struct gaih_servtuple *)
		alloca_account (sizeof (struct gaih_servtuple), alloca_used);

	      int rc = gaih_inet_serv (service->name, tp, req, st, tmpbuf);
	      if (__glibc_unlikely (rc != 0))
		return rc;
	    }
	  else
	    {
	      struct gaih_servtuple **pst = &st;
	      for (tp++; tp->name[0]; tp++)
		{
		  struct gaih_servtuple *newp;

		  if ((tp->protoflag & GAI_PROTO_NOSERVICE) != 0)
		    continue;

		  if (req->ai_socktype != 0
		      && req->ai_socktype != tp->socktype)
		    continue;
		  if (req->ai_protocol != 0
		      && !(tp->protoflag & GAI_PROTO_PROTOANY)
		      && req->ai_protocol != tp->protocol)
		    continue;

		  newp = (struct gaih_servtuple *)
		    alloca_account (sizeof (struct gaih_servtuple),
				    alloca_used);

		  if (gaih_inet_serv (service->name,
				      tp, req, newp, tmpbuf) != 0)
		    continue;

		  *pst = newp;
		  pst = &(newp->next);
		}
	      if (st == (struct gaih_servtuple *) &nullserv)
		return -EAI_SERVICE;
	    }
	}
      else
	{
	  port = htons (service->num);
	  goto got_port;
	}
    }
  else
    {
    got_port:

      if (req->ai_socktype || req->ai_protocol)
	{
	  st = alloca_account (sizeof (struct gaih_servtuple), alloca_used);
	  st->next = NULL;
	  st->socktype = tp->socktype;
	  st->protocol = ((tp->protoflag & GAI_PROTO_PROTOANY)
			  ? req->ai_protocol : tp->protocol);
	  st->port = port;
	}
      else
	{
	  /* Neither socket type nor protocol is set.  Return all socket types
	     we know about.  */
	  struct gaih_servtuple **lastp = &st;
	  for (++tp; tp->name[0]; ++tp)
	    if (tp->defaultflag)
	      {
		struct gaih_servtuple *newp;

		newp = alloca_account (sizeof (struct gaih_servtuple),
				       alloca_used);
		newp->next = NULL;
		newp->socktype = tp->socktype;
		newp->protocol = tp->protocol;
		newp->port = port;

		*lastp = newp;
		lastp = &newp->next;
	      }
	}
    }

  bool malloc_name = false;
  struct gaih_addrtuple *addrmem = NULL;
  char *canonbuf = NULL;
  int result = 0;

  if (name != NULL)
    {
      at = alloca_account (sizeof (struct gaih_addrtuple), alloca_used);
      at->family = AF_UNSPEC;
      at->scopeid = 0;
      at->next = NULL;

      if (req->ai_flags & AI_IDN)
	{
	  char *out;
	  result = __idna_to_dns_encoding (name, &out);
	  if (result != 0)
	    return -result;
	  name = out;
	  malloc_name = true;
	}

      if (__inet_aton_exact (name, (struct in_addr *) at->addr) != 0)
	{
	  if (req->ai_family == AF_UNSPEC || req->ai_family == AF_INET)
	    at->family = AF_INET;
	  else if (req->ai_family == AF_INET6 && (req->ai_flags & AI_V4MAPPED))
	    {
	      at->addr[3] = at->addr[0];
	      at->addr[2] = htonl (0xffff);
	      at->addr[1] = 0;
	      at->addr[0] = 0;
	      at->family = AF_INET6;
	    }
	  else
	    {
	      result = -EAI_ADDRFAMILY;
	      goto free_and_return;
	    }

	  if (req->ai_flags & AI_CANONNAME)
	    canon = name;
	}
      else if (at->family == AF_UNSPEC)
	{
	  char *scope_delim = strchr (name, SCOPE_DELIMITER);
	  int e;
	  if (scope_delim == NULL)
	    e = inet_pton (AF_INET6, name, at->addr);
	  else
	    e = __inet_pton_length (AF_INET6, name, scope_delim - name,
				    at->addr);
	  if (e > 0)
	    {
	      if (req->ai_family == AF_UNSPEC || req->ai_family == AF_INET6)
		at->family = AF_INET6;
	      else if (req->ai_family == AF_INET
		       && IN6_IS_ADDR_V4MAPPED (at->addr))
		{
		  at->addr[0] = at->addr[3];
		  at->family = AF_INET;
		}
	      else
		{
		  result = -EAI_ADDRFAMILY;
		  goto free_and_return;
		}

	      if (scope_delim != NULL
		  && __inet6_scopeid_pton ((struct in6_addr *) at->addr,
					   scope_delim + 1,
					   &at->scopeid) != 0)
		{
		  result = -EAI_NONAME;
		  goto free_and_return;
		}

	      if (req->ai_flags & AI_CANONNAME)
		canon = name;
	    }
	}

      if (at->family == AF_UNSPEC && (req->ai_flags & AI_NUMERICHOST) == 0)
	{
	  struct gaih_addrtuple **pat = &at;
	  int no_data = 0;
	  int no_inet6_data = 0;
	  nss_action_list nip;
	  enum nss_status inet6_status = NSS_STATUS_UNAVAIL;
	  enum nss_status status = NSS_STATUS_UNAVAIL;
	  int no_more;
	  struct resolv_context *res_ctx = NULL;

	  /* If we do not have to look for IPv6 addresses or the canonical
	     name, use the simple, old functions, which do not support
	     IPv6 scope ids, nor retrieving the canonical name.  */
	  if (req->ai_family == AF_INET
	      && (req->ai_flags & AI_CANONNAME) == 0)
	    {
	      int rc;
	      struct hostent th;
	      struct hostent *h;

	      while (1)
		{
		  rc = __gethostbyname2_r (name, AF_INET, &th,
					   tmpbuf->data, tmpbuf->length,
					   &h, &h_errno);
		  if (rc != ERANGE || h_errno != NETDB_INTERNAL)
		    break;
		  if (!scratch_buffer_grow (tmpbuf))
		    {
		      result = -EAI_MEMORY;
		      goto free_and_return;
		    }
		}

	      if (rc == 0)
		{
		  if (h != NULL)
		    {
		      /* We found data, convert it.  */
		      if (!convert_hostent_to_gaih_addrtuple
			  (req, AF_INET, h, &addrmem))
			{
			  result = -EAI_MEMORY;
			  goto free_and_return;
			}
		      *pat = addrmem;
		    }
		  else
		    {
		      if (h_errno == NO_DATA)
			result = -EAI_NODATA;
		      else
			result = -EAI_NONAME;
		      goto free_and_return;
		    }
		}
	      else
		{
		  if (h_errno == NETDB_INTERNAL)
		    result = -EAI_SYSTEM;
		  else if (h_errno == TRY_AGAIN)
		    result = -EAI_AGAIN;
		  else
		    /* We made requests but they turned out no data.
		       The name is known, though.  */
		    result = -EAI_NODATA;

		  goto free_and_return;
		}

	      goto process_list;
	    }

#ifdef USE_NSCD
	  if (__nss_not_use_nscd_hosts > 0
	      && ++__nss_not_use_nscd_hosts > NSS_NSCD_RETRY)
	    __nss_not_use_nscd_hosts = 0;

	  if (!__nss_not_use_nscd_hosts
	      && !__nss_database_custom[NSS_DBSIDX_hosts])
	    {
	      /* Try to use nscd.  */
	      struct nscd_ai_result *air = NULL;
	      int err = __nscd_getai (name, &air, &h_errno);
	      if (air != NULL)
		{
		  /* Transform into gaih_addrtuple list.  */
		  bool added_canon = (req->ai_flags & AI_CANONNAME) == 0;
		  char *addrs = air->addrs;

		  addrmem = calloc (air->naddrs, sizeof (*addrmem));
		  if (addrmem == NULL)
		    {
		      result = -EAI_MEMORY;
		      goto free_and_return;
		    }

		  struct gaih_addrtuple *addrfree = addrmem;
		  for (int i = 0; i < air->naddrs; ++i)
		    {
		      socklen_t size = (air->family[i] == AF_INET
					? INADDRSZ : IN6ADDRSZ);

		      if (!((air->family[i] == AF_INET
			     && req->ai_family == AF_INET6
			     && (req->ai_flags & AI_V4MAPPED) != 0)
			    || req->ai_family == AF_UNSPEC
			    || air->family[i] == req->ai_family))
			{
			  /* Skip over non-matching result.  */
			  addrs += size;
			  continue;
			}

		      if (*pat == NULL)
			{
			  *pat = addrfree++;
			  (*pat)->scopeid = 0;
			}
		      uint32_t *pataddr = (*pat)->addr;
		      (*pat)->next = NULL;
		      if (added_canon || air->canon == NULL)
			(*pat)->name = NULL;
		      else if (canonbuf == NULL)
			{
			  canonbuf = __strdup (air->canon);
			  if (canonbuf == NULL)
			    {
			      result = -EAI_MEMORY;
			      goto free_and_return;
			    }
			  canon = (*pat)->name = canonbuf;
			}

		      if (air->family[i] == AF_INET
			  && req->ai_family == AF_INET6
			  && (req->ai_flags & AI_V4MAPPED))
			{
			  (*pat)->family = AF_INET6;
			  pataddr[3] = *(uint32_t *) addrs;
			  pataddr[2] = htonl (0xffff);
			  pataddr[1] = 0;
			  pataddr[0] = 0;
			  pat = &((*pat)->next);
			  added_canon = true;
			}
		      else if (req->ai_family == AF_UNSPEC
			       || air->family[i] == req->ai_family)
			{
			  (*pat)->family = air->family[i];
			  memcpy (pataddr, addrs, size);
			  pat = &((*pat)->next);
			  added_canon = true;
			  if (air->family[i] == AF_INET6)
			    got_ipv6 = true;
			}
		      addrs += size;
		    }

		  free (air);

		  if (at->family == AF_UNSPEC)
		    {
		      result = -EAI_NONAME;
		      goto free_and_return;
		    }

		  goto process_list;
		}
	      else if (err == 0)
		/* The database contains a negative entry.  */
		goto free_and_return;
	      else if (__nss_not_use_nscd_hosts == 0)
		{
		  if (h_errno == NETDB_INTERNAL && errno == ENOMEM)
		    result = -EAI_MEMORY;
		  else if (h_errno == TRY_AGAIN)
		    result = -EAI_AGAIN;
		  else
		    result = -EAI_SYSTEM;

		  goto free_and_return;
		}
	    }
#endif

	  no_more = !__nss_database_get (nss_database_hosts, &nip);

	  /* If we are looking for both IPv4 and IPv6 address we don't
	     want the lookup functions to automatically promote IPv4
	     addresses to IPv6 addresses, so we use the no_inet6
	     function variant.  */
	  res_ctx = __resolv_context_get ();
	  if (res_ctx == NULL)
	    no_more = 1;

	  while (!no_more)
	    {
	      no_data = 0;
	      nss_gethostbyname4_r *fct4 = NULL;

	      /* gethostbyname4_r sends out parallel A and AAAA queries and
		 is thus only suitable for PF_UNSPEC.  */
	      if (req->ai_family == PF_UNSPEC)
		fct4 = __nss_lookup_function (nip, "gethostbyname4_r");

	      if (fct4 != NULL)
		{
		  while (1)
		    {
		      status = DL_CALL_FCT (fct4, (name, pat,
						   tmpbuf->data, tmpbuf->length,
						   &errno, &h_errno,
						   NULL));
		      if (status == NSS_STATUS_SUCCESS)
			break;
		      if (status != NSS_STATUS_TRYAGAIN
			  || errno != ERANGE || h_errno != NETDB_INTERNAL)
			{
			  if (h_errno == TRY_AGAIN)
			    no_data = EAI_AGAIN;
			  else
			    no_data = h_errno == NO_DATA;
			  break;
			}

		      if (!scratch_buffer_grow (tmpbuf))
			{
			  __resolv_context_put (res_ctx);
			  result = -EAI_MEMORY;
			  goto free_and_return;
			}
		    }

		  if (status == NSS_STATUS_SUCCESS)
		    {
		      assert (!no_data);
		      no_data = 1;

		      if ((req->ai_flags & AI_CANONNAME) != 0 && canon == NULL)
			canon = (*pat)->name;

		      while (*pat != NULL)
			{
			  if ((*pat)->family == AF_INET
			      && req->ai_family == AF_INET6
			      && (req->ai_flags & AI_V4MAPPED) != 0)
			    {
			      uint32_t *pataddr = (*pat)->addr;
			      (*pat)->family = AF_INET6;
			      pataddr[3] = pataddr[0];
			      pataddr[2] = htonl (0xffff);
			      pataddr[1] = 0;
			      pataddr[0] = 0;
			      pat = &((*pat)->next);
			      no_data = 0;
			    }
			  else if (req->ai_family == AF_UNSPEC
				   || (*pat)->family == req->ai_family)
			    {
			      pat = &((*pat)->next);

			      no_data = 0;
			      if (req->ai_family == AF_INET6)
				got_ipv6 = true;
			    }
			  else
			    *pat = ((*pat)->next);
			}
		    }

		  no_inet6_data = no_data;
		}
	      else
		{
		  nss_gethostbyname3_r *fct = NULL;
		  if (req->ai_flags & AI_CANONNAME)
		    /* No need to use this function if we do not look for
		       the canonical name.  The function does not exist in
		       all NSS modules and therefore the lookup would
		       often fail.  */
		    fct = __nss_lookup_function (nip, "gethostbyname3_r");
		  if (fct == NULL)
		    /* We are cheating here.  The gethostbyname2_r
		       function does not have the same interface as
		       gethostbyname3_r but the extra arguments the
		       latter takes are added at the end.  So the
		       gethostbyname2_r code will just ignore them.  */
		    fct = __nss_lookup_function (nip, "gethostbyname2_r");

		  if (fct != NULL)
		    {
		      if (req->ai_family == AF_INET6
			  || req->ai_family == AF_UNSPEC)
			{
			  gethosts (AF_INET6, struct in6_addr);
			  no_inet6_data = no_data;
			  inet6_status = status;
			}
		      if (req->ai_family == AF_INET
			  || req->ai_family == AF_UNSPEC
			  || (req->ai_family == AF_INET6
			      && (req->ai_flags & AI_V4MAPPED)
			      /* Avoid generating the mapped addresses if we
				 know we are not going to need them.  */
			      && ((req->ai_flags & AI_ALL) || !got_ipv6)))
			{
			  gethosts (AF_INET, struct in_addr);

			  if (req->ai_family == AF_INET)
			    {
			      no_inet6_data = no_data;
			      inet6_status = status;
			    }
			}

		      /* If we found one address for AF_INET or AF_INET6,
			 don't continue the search.  */
		      if (inet6_status == NSS_STATUS_SUCCESS
			  || status == NSS_STATUS_SUCCESS)
			{
			  if ((req->ai_flags & AI_CANONNAME) != 0
			      && canon == NULL)
			    {
			      canonbuf = getcanonname (nip, at, name);
			      if (canonbuf == NULL)
				{
				  __resolv_context_put (res_ctx);
				  result = -EAI_MEMORY;
				  goto free_and_return;
				}
			      canon = canonbuf;
			    }
			  status = NSS_STATUS_SUCCESS;
			}
		      else
			{
			  /* We can have different states for AF_INET and
			     AF_INET6.  Try to find a useful one for both.  */
			  if (inet6_status == NSS_STATUS_TRYAGAIN)
			    status = NSS_STATUS_TRYAGAIN;
			  else if (status == NSS_STATUS_UNAVAIL
				   && inet6_status != NSS_STATUS_UNAVAIL)
			    status = inet6_status;
			}
		    }
		  else
		    {
		      /* Could not locate any of the lookup functions.
			 The NSS lookup code does not consistently set
			 errno, so we need to supply our own error
			 code here.  The root cause could either be a
			 resource allocation failure, or a missing
			 service function in the DSO (so it should not
			 be listed in /etc/nsswitch.conf).  Assume the
			 former, and return EBUSY.  */
		      status = NSS_STATUS_UNAVAIL;
		     __set_h_errno (NETDB_INTERNAL);
		     __set_errno (EBUSY);
		    }
		}

	      if (nss_next_action (nip, status) == NSS_ACTION_RETURN)
		break;

	      nip++;
	      if (nip->module == NULL)
		no_more = -1;
	    }

	  __resolv_context_put (res_ctx);

	  /* If we have a failure which sets errno, report it using
	     EAI_SYSTEM.  */
	  if ((status == NSS_STATUS_TRYAGAIN || status == NSS_STATUS_UNAVAIL)
	      && h_errno == NETDB_INTERNAL)
	    {
	      result = -EAI_SYSTEM;
	      goto free_and_return;
	    }

	  if (no_data != 0 && no_inet6_data != 0)
	    {
	      /* If both requests timed out report this.  */
	      if (no_data == EAI_AGAIN && no_inet6_data == EAI_AGAIN)
		result = -EAI_AGAIN;
	      else
		/* We made requests but they turned out no data.  The name
		   is known, though.  */
		result = -EAI_NODATA;

	      goto free_and_return;
	    }
	}

    process_list:
      if (at->family == AF_UNSPEC)
	{
	  result = -EAI_NONAME;
	  goto free_and_return;
	}
    }
  else
    {
      struct gaih_addrtuple *atr;
      atr = at = alloca_account (sizeof (struct gaih_addrtuple), alloca_used);
      memset (at, '\0', sizeof (struct gaih_addrtuple));

      if (req->ai_family == AF_UNSPEC)
	{
	  at->next = __alloca (sizeof (struct gaih_addrtuple));
	  memset (at->next, '\0', sizeof (struct gaih_addrtuple));
	}

      if (req->ai_family == AF_UNSPEC || req->ai_family == AF_INET6)
	{
	  at->family = AF_INET6;
	  if ((req->ai_flags & AI_PASSIVE) == 0)
	    memcpy (at->addr, &in6addr_loopback, sizeof (struct in6_addr));
	  atr = at->next;
	}

      if (req->ai_family == AF_UNSPEC || req->ai_family == AF_INET)
	{
	  atr->family = AF_INET;
	  if ((req->ai_flags & AI_PASSIVE) == 0)
	    atr->addr[0] = htonl (INADDR_LOOPBACK);
	}
    }

  {
    struct gaih_servtuple *st2;
    struct gaih_addrtuple *at2 = at;
    size_t socklen;
    sa_family_t family;

    /*
      buffer is the size of an unformatted IPv6 address in printable format.
     */
    while (at2 != NULL)
      {
	/* Only the first entry gets the canonical name.  */
	if (at2 == at && (req->ai_flags & AI_CANONNAME) != 0)
	  {
	    if (canon == NULL)
	      /* If the canonical name cannot be determined, use
		 the passed in string.  */
	      canon = orig_name;

	    bool do_idn = req->ai_flags & AI_CANONIDN;
	    if (do_idn)
	      {
		char *out;
		int rc = __idna_from_dns_encoding (canon, &out);
		if (rc == 0)
		  canon = out;
		else if (rc == EAI_IDN_ENCODE)
		  /* Use the punycode name as a fallback.  */
		  do_idn = false;
		else
		  {
		    result = -rc;
		    goto free_and_return;
		  }
	      }
	    if (!do_idn)
	      {
		if (canonbuf != NULL)
		  /* We already allocated the string using malloc, but
		     the buffer is now owned by canon.  */
		  canonbuf = NULL;
		else
		  {
		    canon = __strdup (canon);
		    if (canon == NULL)
		      {
			result = -EAI_MEMORY;
			goto free_and_return;
		      }
		  }
	      }
	  }

	family = at2->family;
	if (family == AF_INET6)
	  {
	    socklen = sizeof (struct sockaddr_in6);

	    /* If we looked up IPv4 mapped address discard them here if
	       the caller isn't interested in all address and we have
	       found at least one IPv6 address.  */
	    if (got_ipv6
		&& (req->ai_flags & (AI_V4MAPPED|AI_ALL)) == AI_V4MAPPED
		&& IN6_IS_ADDR_V4MAPPED (at2->addr))
	      goto ignore;
	  }
	else
	  socklen = sizeof (struct sockaddr_in);

	for (st2 = st; st2 != NULL; st2 = st2->next)
	  {
	    struct addrinfo *ai;
	    ai = *pai = malloc (sizeof (struct addrinfo) + socklen);
	    if (ai == NULL)
	      {
		free ((char *) canon);
		result = -EAI_MEMORY;
		goto free_and_return;
	      }

	    ai->ai_flags = req->ai_flags;
	    ai->ai_family = family;
	    ai->ai_socktype = st2->socktype;
	    ai->ai_protocol = st2->protocol;
	    ai->ai_addrlen = socklen;
	    ai->ai_addr = (void *) (ai + 1);

	    /* We only add the canonical name once.  */
	    ai->ai_canonname = (char *) canon;
	    canon = NULL;

#ifdef _HAVE_SA_LEN
	    ai->ai_addr->sa_len = socklen;
#endif /* _HAVE_SA_LEN */
	    ai->ai_addr->sa_family = family;

	    /* In case of an allocation error the list must be NULL
	       terminated.  */
	    ai->ai_next = NULL;

	    if (family == AF_INET6)
	      {
		struct sockaddr_in6 *sin6p =
		  (struct sockaddr_in6 *) ai->ai_addr;

		sin6p->sin6_port = st2->port;
		sin6p->sin6_flowinfo = 0;
		memcpy (&sin6p->sin6_addr,
			at2->addr, sizeof (struct in6_addr));
		sin6p->sin6_scope_id = at2->scopeid;
	      }
	    else
	      {
		struct sockaddr_in *sinp =
		  (struct sockaddr_in *) ai->ai_addr;
		sinp->sin_port = st2->port;
		memcpy (&sinp->sin_addr,
			at2->addr, sizeof (struct in_addr));
		memset (sinp->sin_zero, '\0', sizeof (sinp->sin_zero));
	      }

	    pai = &(ai->ai_next);
	  }

	++*naddrs;

      ignore:
	at2 = at2->next;
      }
  }

 free_and_return:
  if (malloc_name)
    free ((char *) name);
  free (addrmem);
  free (canonbuf);

  return result;
}


struct sort_result
{
  struct addrinfo *dest_addr;
  /* Using sockaddr_storage is for now overkill.  We only support IPv4
     and IPv6 so far.  If this changes at some point we can adjust the
     type here.  */
  struct sockaddr_in6 source_addr;
  uint8_t source_addr_len;
  bool got_source_addr;
  uint8_t source_addr_flags;
  uint8_t prefixlen;
  uint32_t index;
  int32_t native;
};

struct sort_result_combo
{
  struct sort_result *results;
  int nresults;
};


#if __BYTE_ORDER == __BIG_ENDIAN
# define htonl_c(n) n
#else
# define htonl_c(n) __bswap_constant_32 (n)
#endif

static const struct scopeentry
{
  union
  {
    char addr[4];
    uint32_t addr32;
  };
  uint32_t netmask;
  int32_t scope;
} default_scopes[] =
  {
    /* Link-local addresses: scope 2.  */
    { { { 169, 254, 0, 0 } }, htonl_c (0xffff0000), 2 },
    { { { 127, 0, 0, 0 } }, htonl_c (0xff000000), 2 },
    /* Default: scope 14.  */
    { { { 0, 0, 0, 0 } }, htonl_c (0x00000000), 14 }
  };

/* The label table.  */
static const struct scopeentry *scopes;


static int
get_scope (const struct sockaddr_in6 *in6)
{
  int scope;
  if (in6->sin6_family == PF_INET6)
    {
      if (! IN6_IS_ADDR_MULTICAST (&in6->sin6_addr))
	{
	  if (IN6_IS_ADDR_LINKLOCAL (&in6->sin6_addr)
	      /* RFC 4291 2.5.3 says that the loopback address is to be
		 treated like a link-local address.  */
	      || IN6_IS_ADDR_LOOPBACK (&in6->sin6_addr))
	    scope = 2;
	  else if (IN6_IS_ADDR_SITELOCAL (&in6->sin6_addr))
	    scope = 5;
	  else
	    /* XXX Is this the correct default behavior?  */
	    scope = 14;
	}
      else
	scope = in6->sin6_addr.s6_addr[1] & 0xf;
    }
  else if (in6->sin6_family == PF_INET)
    {
      const struct sockaddr_in *in = (const struct sockaddr_in *) in6;

      size_t cnt = 0;
      while (1)
	{
	  if ((in->sin_addr.s_addr & scopes[cnt].netmask)
	      == scopes[cnt].addr32)
	    return scopes[cnt].scope;

	  ++cnt;
	}
      /* NOTREACHED */
    }
  else
    /* XXX What is a good default?  */
    scope = 15;

  return scope;
}


struct prefixentry
{
  struct in6_addr prefix;
  unsigned int bits;
  int val;
};


/* The label table.  */
static const struct prefixentry *labels;

/* Default labels.  */
static const struct prefixentry default_labels[] =
  {
    /* See RFC 3484 for the details.  */
    { { .__in6_u
	= { .__u6_addr8 = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01 } }
      }, 128, 0 },
    { { .__in6_u
	= { .__u6_addr8 = { 0x20, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 } }
      }, 16, 2 },
    { { .__in6_u
	= { .__u6_addr8 = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 } }
      }, 96, 3 },
    { { .__in6_u
	= { .__u6_addr8 = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			    0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00 } }
      }, 96, 4 },
    /* The next two entries differ from RFC 3484.  We need to treat
       IPv6 site-local addresses special because they are never NATed,
       unlike site-locale IPv4 addresses.  If this would not happen, on
       machines which have only IPv4 and IPv6 site-local addresses, the
       sorting would prefer the IPv6 site-local addresses, causing
       unnecessary delays when trying to connect to a global IPv6 address
       through a site-local IPv6 address.  */
    { { .__in6_u
	= { .__u6_addr8 = { 0xfe, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 } }
      }, 10, 5 },
    { { .__in6_u
	= { .__u6_addr8 = { 0xfc, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 } }
      }, 7, 6 },
    /* Additional rule for Teredo tunnels.  */
    { { .__in6_u
	= { .__u6_addr8 = { 0x20, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 } }
      }, 32, 7 },
    { { .__in6_u
	= { .__u6_addr8 = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 } }
      }, 0, 1 }
  };


/* The precedence table.  */
static const struct prefixentry *precedence;

/* The default precedences.  */
static const struct prefixentry default_precedence[] =
  {
    /* See RFC 3484 for the details.  */
    { { .__in6_u
	= { .__u6_addr8 = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01 } }
      }, 128, 50 },
    { { .__in6_u
	= { .__u6_addr8 = { 0x20, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 } }
      }, 16, 30 },
    { { .__in6_u
	= { .__u6_addr8 = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 } }
      }, 96, 20 },
    { { .__in6_u
	= { .__u6_addr8 = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			    0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00 } }
      }, 96, 10 },
    { { .__in6_u
	= { .__u6_addr8 = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 } }
      }, 0, 40 }
  };


static int
match_prefix (const struct sockaddr_in6 *in6,
	      const struct prefixentry *list, int default_val)
{
  int idx;
  struct sockaddr_in6 in6_mem;

  if (in6->sin6_family == PF_INET)
    {
      const struct sockaddr_in *in = (const struct sockaddr_in *) in6;

      /* Construct a V4-to-6 mapped address.  */
      in6_mem.sin6_family = PF_INET6;
      in6_mem.sin6_port = in->sin_port;
      in6_mem.sin6_flowinfo = 0;
      memset (&in6_mem.sin6_addr, '\0', sizeof (in6_mem.sin6_addr));
      in6_mem.sin6_addr.s6_addr16[5] = 0xffff;
      in6_mem.sin6_addr.s6_addr32[3] = in->sin_addr.s_addr;
      in6_mem.sin6_scope_id = 0;

      in6 = &in6_mem;
    }
  else if (in6->sin6_family != PF_INET6)
    return default_val;

  for (idx = 0; ; ++idx)
    {
      unsigned int bits = list[idx].bits;
      const uint8_t *mask = list[idx].prefix.s6_addr;
      const uint8_t *val = in6->sin6_addr.s6_addr;

      while (bits >= 8)
	{
	  if (*mask != *val)
	    break;

	  ++mask;
	  ++val;
	  bits -= 8;
	}

      if (bits < 8)
	{
	  if ((*mask & (0xff00 >> bits)) == (*val & (0xff00 >> bits)))
	    /* Match!  */
	    break;
	}
    }

  return list[idx].val;
}


static int
get_label (const struct sockaddr_in6 *in6)
{
  /* XXX What is a good default value?  */
  return match_prefix (in6, labels, INT_MAX);
}


static int
get_precedence (const struct sockaddr_in6 *in6)
{
  /* XXX What is a good default value?  */
  return match_prefix (in6, precedence, 0);
}


/* Find last bit set in a word.  */
static int
fls (uint32_t a)
{
  uint32_t mask;
  int n;
  for (n = 0, mask = 1 << 31; n < 32; mask >>= 1, ++n)
    if ((a & mask) != 0)
      break;
  return n;
}


static int
rfc3484_sort (const void *p1, const void *p2, void *arg)
{
  const size_t idx1 = *(const size_t *) p1;
  const size_t idx2 = *(const size_t *) p2;
  struct sort_result_combo *src = (struct sort_result_combo *) arg;
  struct sort_result *a1 = &src->results[idx1];
  struct sort_result *a2 = &src->results[idx2];

  /* Rule 1: Avoid unusable destinations.
     We have the got_source_addr flag set if the destination is reachable.  */
  if (a1->got_source_addr && ! a2->got_source_addr)
    return -1;
  if (! a1->got_source_addr && a2->got_source_addr)
    return 1;


  /* Rule 2: Prefer matching scope.  Only interesting if both
     destination addresses are IPv6.  */
  int a1_dst_scope
    = get_scope ((struct sockaddr_in6 *) a1->dest_addr->ai_addr);

  int a2_dst_scope
    = get_scope ((struct sockaddr_in6 *) a2->dest_addr->ai_addr);

  if (a1->got_source_addr)
    {
      int a1_src_scope = get_scope (&a1->source_addr);
      int a2_src_scope = get_scope (&a2->source_addr);

      if (a1_dst_scope == a1_src_scope && a2_dst_scope != a2_src_scope)
	return -1;
      if (a1_dst_scope != a1_src_scope && a2_dst_scope == a2_src_scope)
	return 1;
    }


  /* Rule 3: Avoid deprecated addresses.  */
  if (a1->got_source_addr)
    {
      if (!(a1->source_addr_flags & in6ai_deprecated)
	  && (a2->source_addr_flags & in6ai_deprecated))
	return -1;
      if ((a1->source_addr_flags & in6ai_deprecated)
	  && !(a2->source_addr_flags & in6ai_deprecated))
	return 1;
    }

  /* Rule 4: Prefer home addresses.  */
  if (a1->got_source_addr)
    {
      if (!(a1->source_addr_flags & in6ai_homeaddress)
	  && (a2->source_addr_flags & in6ai_homeaddress))
	return 1;
      if ((a1->source_addr_flags & in6ai_homeaddress)
	  && !(a2->source_addr_flags & in6ai_homeaddress))
	return -1;
    }

  /* Rule 5: Prefer matching label.  */
  if (a1->got_source_addr)
    {
      int a1_dst_label
	= get_label ((struct sockaddr_in6 *) a1->dest_addr->ai_addr);
      int a1_src_label = get_label (&a1->source_addr);

      int a2_dst_label
	= get_label ((struct sockaddr_in6 *) a2->dest_addr->ai_addr);
      int a2_src_label = get_label (&a2->source_addr);

      if (a1_dst_label == a1_src_label && a2_dst_label != a2_src_label)
	return -1;
      if (a1_dst_label != a1_src_label && a2_dst_label == a2_src_label)
	return 1;
    }


  /* Rule 6: Prefer higher precedence.  */
  int a1_prec
    = get_precedence ((struct sockaddr_in6 *) a1->dest_addr->ai_addr);
  int a2_prec
    = get_precedence ((struct sockaddr_in6 *) a2->dest_addr->ai_addr);

  if (a1_prec > a2_prec)
    return -1;
  if (a1_prec < a2_prec)
    return 1;


  /* Rule 7: Prefer native transport.  */
  if (a1->got_source_addr)
    {
      /* The same interface index means the same interface which means
	 there is no difference in transport.  This should catch many
	 (most?) cases.  */
      if (a1->index != a2->index)
	{
	  int a1_native = a1->native;
	  int a2_native = a2->native;

	  if (a1_native == -1 || a2_native == -1)
	    {
	      uint32_t a1_index;
	      if (a1_native == -1)
		{
		  /* If we do not have the information use 'native' as
		     the default.  */
		  a1_native = 0;
		  a1_index = a1->index;
		}
	      else
		a1_index = 0xffffffffu;

	      uint32_t a2_index;
	      if (a2_native == -1)
		{
		  /* If we do not have the information use 'native' as
		     the default.  */
		  a2_native = 0;
		  a2_index = a2->index;
		}
	      else
		a2_index = 0xffffffffu;

	      __check_native (a1_index, &a1_native, a2_index, &a2_native);

	      /* Fill in the results in all the records.  */
	      for (int i = 0; i < src->nresults; ++i)
		if (a1_index != -1 && src->results[i].index == a1_index)
		  {
		    assert (src->results[i].native == -1
			    || src->results[i].native == a1_native);
		    src->results[i].native = a1_native;
		  }
		else if (a2_index != -1 && src->results[i].index == a2_index)
		  {
		    assert (src->results[i].native == -1
			    || src->results[i].native == a2_native);
		    src->results[i].native = a2_native;
		  }
	    }

	  if (a1_native && !a2_native)
	    return -1;
	  if (!a1_native && a2_native)
	    return 1;
	}
    }


  /* Rule 8: Prefer smaller scope.  */
  if (a1_dst_scope < a2_dst_scope)
    return -1;
  if (a1_dst_scope > a2_dst_scope)
    return 1;


  /* Rule 9: Use longest matching prefix.  */
  if (a1->got_source_addr
      && a1->dest_addr->ai_family == a2->dest_addr->ai_family)
    {
      int bit1 = 0;
      int bit2 = 0;

      if (a1->dest_addr->ai_family == PF_INET)
	{
	  assert (a1->source_addr.sin6_family == PF_INET);
	  assert (a2->source_addr.sin6_family == PF_INET);

	  /* Outside of subnets, as defined by the network masks,
	     common address prefixes for IPv4 addresses make no sense.
	     So, define a non-zero value only if source and
	     destination address are on the same subnet.  */
	  struct sockaddr_in *in1_dst
	    = (struct sockaddr_in *) a1->dest_addr->ai_addr;
	  in_addr_t in1_dst_addr = ntohl (in1_dst->sin_addr.s_addr);
	  struct sockaddr_in *in1_src
	    = (struct sockaddr_in *) &a1->source_addr;
	  in_addr_t in1_src_addr = ntohl (in1_src->sin_addr.s_addr);
	  in_addr_t netmask1 = 0xffffffffu << (32 - a1->prefixlen);

	  if ((in1_src_addr & netmask1) == (in1_dst_addr & netmask1))
	    bit1 = fls (in1_dst_addr ^ in1_src_addr);

	  struct sockaddr_in *in2_dst
	    = (struct sockaddr_in *) a2->dest_addr->ai_addr;
	  in_addr_t in2_dst_addr = ntohl (in2_dst->sin_addr.s_addr);
	  struct sockaddr_in *in2_src
	    = (struct sockaddr_in *) &a2->source_addr;
	  in_addr_t in2_src_addr = ntohl (in2_src->sin_addr.s_addr);
	  in_addr_t netmask2 = 0xffffffffu << (32 - a2->prefixlen);

	  if ((in2_src_addr & netmask2) == (in2_dst_addr & netmask2))
	    bit2 = fls (in2_dst_addr ^ in2_src_addr);
	}
      else if (a1->dest_addr->ai_family == PF_INET6)
	{
	  assert (a1->source_addr.sin6_family == PF_INET6);
	  assert (a2->source_addr.sin6_family == PF_INET6);

	  struct sockaddr_in6 *in1_dst;
	  struct sockaddr_in6 *in1_src;
	  struct sockaddr_in6 *in2_dst;
	  struct sockaddr_in6 *in2_src;

	  in1_dst = (struct sockaddr_in6 *) a1->dest_addr->ai_addr;
	  in1_src = (struct sockaddr_in6 *) &a1->source_addr;
	  in2_dst = (struct sockaddr_in6 *) a2->dest_addr->ai_addr;
	  in2_src = (struct sockaddr_in6 *) &a2->source_addr;

	  int i;
	  for (i = 0; i < 4; ++i)
	    if (in1_dst->sin6_addr.s6_addr32[i]
		!= in1_src->sin6_addr.s6_addr32[i]
		|| (in2_dst->sin6_addr.s6_addr32[i]
		    != in2_src->sin6_addr.s6_addr32[i]))
	      break;

	  if (i < 4)
	    {
	      bit1 = fls (ntohl (in1_dst->sin6_addr.s6_addr32[i]
				 ^ in1_src->sin6_addr.s6_addr32[i]));
	      bit2 = fls (ntohl (in2_dst->sin6_addr.s6_addr32[i]
				 ^ in2_src->sin6_addr.s6_addr32[i]));
	    }
	}

      if (bit1 > bit2)
	return -1;
      if (bit1 < bit2)
	return 1;
    }


  /* Rule 10: Otherwise, leave the order unchanged.  To ensure this
     compare with the value indicating the order in which the entries
     have been received from the services.  NB: no two entries can have
     the same order so the test will never return zero.  */
  return idx1 < idx2 ? -1 : 1;
}


static int
in6aicmp (const void *p1, const void *p2)
{
  struct in6addrinfo *a1 = (struct in6addrinfo *) p1;
  struct in6addrinfo *a2 = (struct in6addrinfo *) p2;

  return memcmp (a1->addr, a2->addr, sizeof (a1->addr));
}


/* Name of the config file for RFC 3484 sorting (for now).  */
#define GAICONF_FNAME "/etc/gai.conf"


/* Non-zero if we are supposed to reload the config file automatically
   whenever it changed.  */
static int gaiconf_reload_flag;

/* Non-zero if gaiconf_reload_flag was ever set to true.  */
static int gaiconf_reload_flag_ever_set;

/* Last modification time.  */
#ifdef _STATBUF_ST_NSEC

static struct __timespec64 gaiconf_mtime;

static inline void
save_gaiconf_mtime (const struct __stat64_t64 *st)
{
  gaiconf_mtime = (struct __timespec64) { st->st_mtim.tv_sec,
					  st->st_mtim.tv_nsec };
}

static inline bool
check_gaiconf_mtime (const struct __stat64_t64 *st)
{
  return (st->st_mtim.tv_sec == gaiconf_mtime.tv_sec
          && st->st_mtim.tv_nsec == gaiconf_mtime.tv_nsec);
}

#else

static time_t gaiconf_mtime;

static inline void
save_gaiconf_mtime (const struct __stat64_t64 *st)
{
  gaiconf_mtime = st->st_mtime;
}

static inline bool
check_gaiconf_mtime (const struct __stat64_t64 *st)
{
  return st->st_mtime == gaiconf_mtime;
}

#endif


libc_freeres_fn(fini)
{
  if (labels != default_labels)
    {
      const struct prefixentry *old = labels;
      labels = default_labels;
      free ((void *) old);
    }

  if (precedence != default_precedence)
    {
      const struct prefixentry *old = precedence;
      precedence = default_precedence;
      free ((void *) old);
    }

  if (scopes != default_scopes)
    {
      const struct scopeentry *old = scopes;
      scopes = default_scopes;
      free ((void *) old);
    }
}


struct prefixlist
{
  struct prefixentry entry;
  struct prefixlist *next;
};


struct scopelist
{
  struct scopeentry entry;
  struct scopelist *next;
};


static void
free_prefixlist (struct prefixlist *list)
{
  while (list != NULL)
    {
      struct prefixlist *oldp = list;
      list = list->next;
      free (oldp);
    }
}


static void
free_scopelist (struct scopelist *list)
{
  while (list != NULL)
    {
      struct scopelist *oldp = list;
      list = list->next;
      free (oldp);
    }
}


static int
prefixcmp (const void *p1, const void *p2)
{
  const struct prefixentry *e1 = (const struct prefixentry *) p1;
  const struct prefixentry *e2 = (const struct prefixentry *) p2;

  if (e1->bits < e2->bits)
    return 1;
  if (e1->bits == e2->bits)
    return 0;
  return -1;
}


static int
scopecmp (const void *p1, const void *p2)
{
  const struct scopeentry *e1 = (const struct scopeentry *) p1;
  const struct scopeentry *e2 = (const struct scopeentry *) p2;

  if (e1->netmask > e2->netmask)
    return -1;
  if (e1->netmask == e2->netmask)
    return 0;
  return 1;
}


static void
gaiconf_init (void)
{
  struct prefixlist *labellist = NULL;
  size_t nlabellist = 0;
  bool labellist_nullbits = false;
  struct prefixlist *precedencelist = NULL;
  size_t nprecedencelist = 0;
  bool precedencelist_nullbits = false;
  struct scopelist *scopelist =  NULL;
  size_t nscopelist = 0;
  bool scopelist_nullbits = false;

  FILE *fp = fopen (GAICONF_FNAME, "rce");
  if (fp != NULL)
    {
      struct __stat64_t64 st;
      if (__fstat64_time64 (fileno (fp), &st) != 0)
	{
	  fclose (fp);
	  goto no_file;
	}

      char *line = NULL;
      size_t linelen = 0;

      __fsetlocking (fp, FSETLOCKING_BYCALLER);

      while (!feof_unlocked (fp))
	{
	  ssize_t n = __getline (&line, &linelen, fp);
	  if (n <= 0)
	    break;

	  /* Handle comments.  No escaping possible so this is easy.  */
	  char *cp = strchr (line, '#');
	  if (cp != NULL)
	    *cp = '\0';

	  cp = line;
	  while (isspace (*cp))
	    ++cp;

	  char *cmd = cp;
	  while (*cp != '\0' && !isspace (*cp))
	    ++cp;
	  size_t cmdlen = cp - cmd;

	  if (*cp != '\0')
	    *cp++ = '\0';
	  while (isspace (*cp))
	    ++cp;

	  char *val1 = cp;
	  while (*cp != '\0' && !isspace (*cp))
	    ++cp;
	  size_t val1len = cp - cmd;

	  /* We always need at least two values.  */
	  if (val1len == 0)
	    continue;

	  if (*cp != '\0')
	    *cp++ = '\0';
	  while (isspace (*cp))
	    ++cp;

	  char *val2 = cp;
	  while (*cp != '\0' && !isspace (*cp))
	    ++cp;

	  /*  Ignore the rest of the line.  */
	  *cp = '\0';

	  struct prefixlist **listp;
	  size_t *lenp;
	  bool *nullbitsp;
	  switch (cmdlen)
	    {
	    case 5:
	      if (strcmp (cmd, "label") == 0)
		{
		  struct in6_addr prefix;
		  unsigned long int bits;
		  unsigned long int val;
		  char *endp;

		  listp = &labellist;
		  lenp = &nlabellist;
		  nullbitsp = &labellist_nullbits;

		new_elem:
		  bits = 128;
		  __set_errno (0);
		  cp = strchr (val1, '/');
		  if (cp != NULL)
		    *cp++ = '\0';
		  if (inet_pton (AF_INET6, val1, &prefix)
		      && (cp == NULL
			  || (bits = strtoul (cp, &endp, 10)) != ULONG_MAX
			  || errno != ERANGE)
		      && *endp == '\0'
		      && bits <= 128
		      && ((val = strtoul (val2, &endp, 10)) != ULONG_MAX
			  || errno != ERANGE)
		      && *endp == '\0'
		      && val <= INT_MAX)
		    {
		      struct prefixlist *newp = malloc (sizeof (*newp));
		      if (newp == NULL)
			{
			  free (line);
			  fclose (fp);
			  goto no_file;
			}

		      memcpy (&newp->entry.prefix, &prefix, sizeof (prefix));
		      newp->entry.bits = bits;
		      newp->entry.val = val;
		      newp->next = *listp;
		      *listp = newp;
		      ++*lenp;
		      *nullbitsp |= bits == 0;
		    }
		}
	      break;

	    case 6:
	      if (strcmp (cmd, "reload") == 0)
		{
		  gaiconf_reload_flag = strcmp (val1, "yes") == 0;
		  if (gaiconf_reload_flag)
		    gaiconf_reload_flag_ever_set = 1;
		}
	      break;

	    case 7:
	      if (strcmp (cmd, "scopev4") == 0)
		{
		  struct in6_addr prefix;
		  unsigned long int bits;
		  unsigned long int val;
		  char *endp;

		  bits = 32;
		  __set_errno (0);
		  cp = strchr (val1, '/');
		  if (cp != NULL)
		    *cp++ = '\0';
		  if (inet_pton (AF_INET6, val1, &prefix))
		    {
		      bits = 128;
		      if (IN6_IS_ADDR_V4MAPPED (&prefix)
			  && (cp == NULL
			      || (bits = strtoul (cp, &endp, 10)) != ULONG_MAX
			      || errno != ERANGE)
			  && *endp == '\0'
			  && bits >= 96
			  && bits <= 128
			  && ((val = strtoul (val2, &endp, 10)) != ULONG_MAX
			      || errno != ERANGE)
			  && *endp == '\0'
			  && val <= INT_MAX)
			{
			  struct scopelist *newp;
			new_scope:
			  newp = malloc (sizeof (*newp));
			  if (newp == NULL)
			    {
			      free (line);
			      fclose (fp);
			      goto no_file;
			    }

			  newp->entry.netmask = htonl (bits != 96
						       ? (0xffffffff
							  << (128 - bits))
						       : 0);
			  newp->entry.addr32 = (prefix.s6_addr32[3]
						& newp->entry.netmask);
			  newp->entry.scope = val;
			  newp->next = scopelist;
			  scopelist = newp;
			  ++nscopelist;
			  scopelist_nullbits |= bits == 96;
			}
		    }
		  else if (inet_pton (AF_INET, val1, &prefix.s6_addr32[3])
			   && (cp == NULL
			       || (bits = strtoul (cp, &endp, 10)) != ULONG_MAX
			       || errno != ERANGE)
			   && *endp == '\0'
			   && bits <= 32
			   && ((val = strtoul (val2, &endp, 10)) != ULONG_MAX
			       || errno != ERANGE)
			   && *endp == '\0'
			   && val <= INT_MAX)
		    {
		      bits += 96;
		      goto new_scope;
		    }
		}
	      break;

	    case 10:
	      if (strcmp (cmd, "precedence") == 0)
		{
		  listp = &precedencelist;
		  lenp = &nprecedencelist;
		  nullbitsp = &precedencelist_nullbits;
		  goto new_elem;
		}
	      break;
	    }
	}

      free (line);

      fclose (fp);

      /* Create the array for the labels.  */
      struct prefixentry *new_labels;
      if (nlabellist > 0)
	{
	  if (!labellist_nullbits)
	    ++nlabellist;
	  new_labels = malloc (nlabellist * sizeof (*new_labels));
	  if (new_labels == NULL)
	    goto no_file;

	  int i = nlabellist;
	  if (!labellist_nullbits)
	    {
	      --i;
	      memset (&new_labels[i].prefix, '\0', sizeof (struct in6_addr));
	      new_labels[i].bits = 0;
	      new_labels[i].val = 1;
	    }

	  struct prefixlist *l = labellist;
	  while (i-- > 0)
	    {
	      new_labels[i] = l->entry;
	      l = l->next;
	    }
	  free_prefixlist (labellist);

	  /* Sort the entries so that the most specific ones are at
	     the beginning.  */
	  qsort (new_labels, nlabellist, sizeof (*new_labels), prefixcmp);
	}
      else
	new_labels = (struct prefixentry *) default_labels;

      struct prefixentry *new_precedence;
      if (nprecedencelist > 0)
	{
	  if (!precedencelist_nullbits)
	    ++nprecedencelist;
	  new_precedence = malloc (nprecedencelist * sizeof (*new_precedence));
	  if (new_precedence == NULL)
	    {
	      if (new_labels != default_labels)
		free (new_labels);
	      goto no_file;
	    }

	  int i = nprecedencelist;
	  if (!precedencelist_nullbits)
	    {
	      --i;
	      memset (&new_precedence[i].prefix, '\0',
		      sizeof (struct in6_addr));
	      new_precedence[i].bits = 0;
	      new_precedence[i].val = 40;
	    }

	  struct prefixlist *l = precedencelist;
	  while (i-- > 0)
	    {
	      new_precedence[i] = l->entry;
	      l = l->next;
	    }
	  free_prefixlist (precedencelist);

	  /* Sort the entries so that the most specific ones are at
	     the beginning.  */
	  qsort (new_precedence, nprecedencelist, sizeof (*new_precedence),
		 prefixcmp);
	}
      else
	new_precedence = (struct prefixentry *) default_precedence;

      struct scopeentry *new_scopes;
      if (nscopelist > 0)
	{
	  if (!scopelist_nullbits)
	    ++nscopelist;
	  new_scopes = malloc (nscopelist * sizeof (*new_scopes));
	  if (new_scopes == NULL)
	    {
	      if (new_labels != default_labels)
		free (new_labels);
	      if (new_precedence != default_precedence)
		free (new_precedence);
	      goto no_file;
	    }

	  int i = nscopelist;
	  if (!scopelist_nullbits)
	    {
	      --i;
	      new_scopes[i].addr32 = 0;
	      new_scopes[i].netmask = 0;
	      new_scopes[i].scope = 14;
	    }

	  struct scopelist *l = scopelist;
	  while (i-- > 0)
	    {
	      new_scopes[i] = l->entry;
	      l = l->next;
	    }
	  free_scopelist (scopelist);

	  /* Sort the entries so that the most specific ones are at
	     the beginning.  */
	  qsort (new_scopes, nscopelist, sizeof (*new_scopes),
		 scopecmp);
	}
      else
	new_scopes = (struct scopeentry *) default_scopes;

      /* Now we are ready to replace the values.  */
      const struct prefixentry *old = labels;
      labels = new_labels;
      if (old != default_labels)
	free ((void *) old);

      old = precedence;
      precedence = new_precedence;
      if (old != default_precedence)
	free ((void *) old);

      const struct scopeentry *oldscope = scopes;
      scopes = new_scopes;
      if (oldscope != default_scopes)
	free ((void *) oldscope);

      save_gaiconf_mtime (&st);
    }
  else
    {
    no_file:
      free_prefixlist (labellist);
      free_prefixlist (precedencelist);
      free_scopelist (scopelist);

      /* If we previously read the file but it is gone now, free the
	 old data and use the builtin one.  Leave the reload flag
	 alone.  */
      fini ();
    }
}


static void
gaiconf_reload (void)
{
  struct __stat64_t64 st;
  if (__stat64_time64 (GAICONF_FNAME, &st) != 0
      || !check_gaiconf_mtime (&st))
    gaiconf_init ();
}


int
getaddrinfo (const char *name, const char *service,
	     const struct addrinfo *hints, struct addrinfo **pai)
{
  int i = 0, last_i = 0;
  int nresults = 0;
  struct addrinfo *p = NULL;
  struct gaih_service gaih_service, *pservice;
  struct addrinfo local_hints;

  if (name != NULL && name[0] == '*' && name[1] == 0)
    name = NULL;

  if (service != NULL && service[0] == '*' && service[1] == 0)
    service = NULL;

  if (name == NULL && service == NULL)
    return EAI_NONAME;

  if (hints == NULL)
    hints = &default_hints;

  if (hints->ai_flags
      & ~(AI_PASSIVE|AI_CANONNAME|AI_NUMERICHOST|AI_ADDRCONFIG|AI_V4MAPPED
	  |AI_IDN|AI_CANONIDN|DEPRECATED_AI_IDN
	  |AI_NUMERICSERV|AI_ALL))
    return EAI_BADFLAGS;

  if ((hints->ai_flags & AI_CANONNAME) && name == NULL)
    return EAI_BADFLAGS;

  if (hints->ai_family != AF_UNSPEC && hints->ai_family != AF_INET
      && hints->ai_family != AF_INET6)
    return EAI_FAMILY;

  struct in6addrinfo *in6ai = NULL;
  size_t in6ailen = 0;
  bool seen_ipv4 = false;
  bool seen_ipv6 = false;
  bool check_pf_called = false;

  if (hints->ai_flags & AI_ADDRCONFIG)
    {
      /* We might need information about what interfaces are available.
	 Also determine whether we have IPv4 or IPv6 interfaces or both.  We
	 cannot cache the results since new interfaces could be added at
	 any time.  */
      __check_pf (&seen_ipv4, &seen_ipv6, &in6ai, &in6ailen);
      check_pf_called = true;

      /* Now make a decision on what we return, if anything.  */
      if (hints->ai_family == PF_UNSPEC && (seen_ipv4 || seen_ipv6))
	{
	  /* If we haven't seen both IPv4 and IPv6 interfaces we can
	     narrow down the search.  */
	  if (seen_ipv4 != seen_ipv6)
	    {
	      local_hints = *hints;
	      local_hints.ai_family = seen_ipv4 ? PF_INET : PF_INET6;
	      hints = &local_hints;
	    }
	}
      else if ((hints->ai_family == PF_INET && ! seen_ipv4)
	       || (hints->ai_family == PF_INET6 && ! seen_ipv6))
	{
	  /* We cannot possibly return a valid answer.  */
	  __free_in6ai (in6ai);
	  return EAI_NONAME;
	}
    }

  if (service && service[0])
    {
      char *c;
      gaih_service.name = service;
      gaih_service.num = strtoul (gaih_service.name, &c, 10);
      if (*c != '\0')
	{
	  if (hints->ai_flags & AI_NUMERICSERV)
	    {
	      __free_in6ai (in6ai);
	      return EAI_NONAME;
	    }

	  gaih_service.num = -1;
	}

      pservice = &gaih_service;
    }
  else
    pservice = NULL;

  struct addrinfo **end = &p;
  unsigned int naddrs = 0;
  struct scratch_buffer tmpbuf;

  scratch_buffer_init (&tmpbuf);
  last_i = gaih_inet (name, pservice, hints, end, &naddrs, &tmpbuf);
  scratch_buffer_free (&tmpbuf);

  if (last_i != 0)
    {
      freeaddrinfo (p);
      __free_in6ai (in6ai);

      return -last_i;
    }

  while (*end)
    {
      end = &((*end)->ai_next);
      ++nresults;
    }

  if (naddrs > 1)
    {
      /* Read the config file.  */
      __libc_once_define (static, once);
      __typeof (once) old_once = once;
      __libc_once (once, gaiconf_init);
      /* Sort results according to RFC 3484.  */
      struct sort_result *results;
      size_t *order;
      struct addrinfo *q;
      struct addrinfo *last = NULL;
      char *canonname = NULL;
      bool malloc_results;
      size_t alloc_size = nresults * (sizeof (*results) + sizeof (size_t));

      malloc_results
	= !__libc_use_alloca (alloc_size);
      if (malloc_results)
	{
	  results = malloc (alloc_size);
	  if (results == NULL)
	    {
	      __free_in6ai (in6ai);
	      return EAI_MEMORY;
	    }
	}
      else
	results = alloca (alloc_size);
      order = (size_t *) (results + nresults);

      /* Now we definitely need the interface information.  */
      if (! check_pf_called)
	__check_pf (&seen_ipv4, &seen_ipv6, &in6ai, &in6ailen);

      /* If we have information about deprecated and temporary addresses
	 sort the array now.  */
      if (in6ai != NULL)
	qsort (in6ai, in6ailen, sizeof (*in6ai), in6aicmp);

      int fd = -1;
      int af = AF_UNSPEC;

      for (i = 0, q = p; q != NULL; ++i, last = q, q = q->ai_next)
	{
	  results[i].dest_addr = q;
	  results[i].native = -1;
	  order[i] = i;

	  /* If we just looked up the address for a different
	     protocol, reuse the result.  */
	  if (last != NULL && last->ai_addrlen == q->ai_addrlen
	      && memcmp (last->ai_addr, q->ai_addr, q->ai_addrlen) == 0)
	    {
	      memcpy (&results[i].source_addr, &results[i - 1].source_addr,
		      results[i - 1].source_addr_len);
	      results[i].source_addr_len = results[i - 1].source_addr_len;
	      results[i].got_source_addr = results[i - 1].got_source_addr;
	      results[i].source_addr_flags = results[i - 1].source_addr_flags;
	      results[i].prefixlen = results[i - 1].prefixlen;
	      results[i].index = results[i - 1].index;
	    }
	  else
	    {
	      results[i].got_source_addr = false;
	      results[i].source_addr_flags = 0;
	      results[i].prefixlen = 0;
	      results[i].index = 0xffffffffu;

	      /* We overwrite the type with SOCK_DGRAM since we do not
		 want connect() to connect to the other side.  If we
		 cannot determine the source address remember this
		 fact.  */
	      if (fd == -1 || (af == AF_INET && q->ai_family == AF_INET6))
		{
		  if (fd != -1)
		  close_retry:
		    __close_nocancel_nostatus (fd);
		  af = q->ai_family;
		  fd = __socket (af, SOCK_DGRAM | SOCK_CLOEXEC, IPPROTO_IP);
		}
	      else
		{
		  /* Reset the connection.  */
		  struct sockaddr sa = { .sa_family = AF_UNSPEC };
		  __connect (fd, &sa, sizeof (sa));
		}

	      socklen_t sl = sizeof (results[i].source_addr);
	      if (fd != -1
		  && __connect (fd, q->ai_addr, q->ai_addrlen) == 0
		  && __getsockname (fd,
				    (struct sockaddr *) &results[i].source_addr,
				    &sl) == 0)
		{
		  results[i].source_addr_len = sl;
		  results[i].got_source_addr = true;

		  if (in6ai != NULL)
		    {
		      /* See whether the source address is on the list of
			 deprecated or temporary addresses.  */
		      struct in6addrinfo tmp;

		      if (q->ai_family == AF_INET && af == AF_INET)
			{
			  struct sockaddr_in *sinp
			    = (struct sockaddr_in *) &results[i].source_addr;
			  tmp.addr[0] = 0;
			  tmp.addr[1] = 0;
			  tmp.addr[2] = htonl (0xffff);
			  /* Special case for lo interface, the source address
			     being possibly different than the interface
			     address. */
			  if ((ntohl(sinp->sin_addr.s_addr) & 0xff000000)
			      == 0x7f000000)
			    tmp.addr[3] = htonl(0x7f000001);
			  else
			    tmp.addr[3] = sinp->sin_addr.s_addr;
			}
		      else
			{
			  struct sockaddr_in6 *sin6p
			    = (struct sockaddr_in6 *) &results[i].source_addr;
			  memcpy (tmp.addr, &sin6p->sin6_addr, IN6ADDRSZ);
			}

		      struct in6addrinfo *found
			= bsearch (&tmp, in6ai, in6ailen, sizeof (*in6ai),
				   in6aicmp);
		      if (found != NULL)
			{
			  results[i].source_addr_flags = found->flags;
			  results[i].prefixlen = found->prefixlen;
			  results[i].index = found->index;
			}
		    }

		  if (q->ai_family == AF_INET && af == AF_INET6)
		    {
		      /* We have to convert the address.  The socket is
			 IPv6 and the request is for IPv4.  */
		      struct sockaddr_in6 *sin6
			= (struct sockaddr_in6 *) &results[i].source_addr;
		      struct sockaddr_in *sin
			= (struct sockaddr_in *) &results[i].source_addr;
		      assert (IN6_IS_ADDR_V4MAPPED (sin6->sin6_addr.s6_addr32));
		      sin->sin_family = AF_INET;
		      /* We do not have to initialize sin_port since this
			 fields has the same position and size in the IPv6
			 structure.  */
		      assert (offsetof (struct sockaddr_in, sin_port)
			      == offsetof (struct sockaddr_in6, sin6_port));
		      assert (sizeof (sin->sin_port)
			      == sizeof (sin6->sin6_port));
		      memcpy (&sin->sin_addr,
			      &sin6->sin6_addr.s6_addr32[3], INADDRSZ);
		      results[i].source_addr_len = sizeof (struct sockaddr_in);
		    }
		}
	      else if (errno == EAFNOSUPPORT && af == AF_INET6
		       && q->ai_family == AF_INET)
		/* This could mean IPv6 sockets are IPv6-only.  */
		goto close_retry;
	      else
		/* Just make sure that if we have to process the same
		   address again we do not copy any memory.  */
		results[i].source_addr_len = 0;
	    }

	  /* Remember the canonical name.  */
	  if (q->ai_canonname != NULL)
	    {
	      assert (canonname == NULL);
	      canonname = q->ai_canonname;
	      q->ai_canonname = NULL;
	    }
	}

      if (fd != -1)
	__close_nocancel_nostatus (fd);

      /* We got all the source addresses we can get, now sort using
	 the information.  */
      struct sort_result_combo src
	= { .results = results, .nresults = nresults };
      if (__glibc_unlikely (gaiconf_reload_flag_ever_set))
	{
	  __libc_lock_define_initialized (static, lock);

	  __libc_lock_lock (lock);
	  if (__libc_once_get (old_once) && gaiconf_reload_flag)
	    gaiconf_reload ();
	  __qsort_r (order, nresults, sizeof (order[0]), rfc3484_sort, &src);
	  __libc_lock_unlock (lock);
	}
      else
	__qsort_r (order, nresults, sizeof (order[0]), rfc3484_sort, &src);

      /* Queue the results up as they come out of sorting.  */
      q = p = results[order[0]].dest_addr;
      for (i = 1; i < nresults; ++i)
	q = q->ai_next = results[order[i]].dest_addr;
      q->ai_next = NULL;

      /* Fill in the canonical name into the new first entry.  */
      p->ai_canonname = canonname;

      if (malloc_results)
	free (results);
    }

  __free_in6ai (in6ai);

  if (p)
    {
      *pai = p;
      return 0;
    }

  return last_i ? -last_i : EAI_NONAME;
}
libc_hidden_def (getaddrinfo)

nss_interface_function (getaddrinfo)

void
freeaddrinfo (struct addrinfo *ai)
{
  struct addrinfo *p;

  while (ai != NULL)
    {
      p = ai;
      ai = ai->ai_next;
      free (p->ai_canonname);
      free (p);
    }
}
libc_hidden_def (freeaddrinfo)
