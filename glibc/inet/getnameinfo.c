/* Convert socket address to string using Name Service Switch modules.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <netdb.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <netinet/in.h>
#include <sys/param.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/utsname.h>
#include <libc-lock.h>
#include <scratch_buffer.h>
#include <net-internal.h>

#ifndef min
# define min(x,y) (((x) > (y)) ? (y) : (x))
#endif /* min */

libc_freeres_ptr (static char *domain);

/* Former NI_IDN_ALLOW_UNASSIGNED, NI_IDN_USE_STD3_ASCII_RULES flags,
   now ignored.  */
#define DEPRECATED_NI_IDN 192

static char *
nrl_domainname (void)
{
  static int not_first;

  if (! not_first)
    {
      __libc_lock_define_initialized (static, lock);
      __libc_lock_lock (lock);

      if (! not_first)
	{
	  char *c;
	  struct hostent *h, th;
	  int herror;
	  struct scratch_buffer tmpbuf;

	  scratch_buffer_init (&tmpbuf);
	  not_first = 1;

	  while (__gethostbyname_r ("localhost", &th,
				    tmpbuf.data, tmpbuf.length,
				    &h, &herror))
	    {
	      if (herror == NETDB_INTERNAL && errno == ERANGE)
		{
		  if (!scratch_buffer_grow (&tmpbuf))
		    goto done;
		}
	      else
		break;
	    }

	  if (h && (c = strchr (h->h_name, '.')))
	    domain = __strdup (++c);
	  else
	    {
	      /* The name contains no domain information.  Use the name
		 now to get more information.  */
	      while (__gethostname (tmpbuf.data, tmpbuf.length))
		if (!scratch_buffer_grow (&tmpbuf))
		  goto done;

	      if ((c = strchr (tmpbuf.data, '.')))
		domain = __strdup (++c);
	      else
		{
		  /* We need to preserve the hostname.  */
		  const char *hstname = strdupa (tmpbuf.data);

		  while (__gethostbyname_r (hstname, &th,
					    tmpbuf.data, tmpbuf.length,
					    &h, &herror))
		    {
		      if (herror == NETDB_INTERNAL && errno == ERANGE)
			{
			  if (!scratch_buffer_grow (&tmpbuf))
			    goto done;
			}
		      else
			break;
		    }

		  if (h && (c = strchr(h->h_name, '.')))
		    domain = __strdup (++c);
		  else
		    {
		      struct in_addr in_addr;

		      in_addr.s_addr = htonl (INADDR_LOOPBACK);

		      while (__gethostbyaddr_r ((const char *) &in_addr,
						sizeof (struct in_addr),
						AF_INET, &th,
						tmpbuf.data, tmpbuf.length,
						&h, &herror))
			{
			  if (herror == NETDB_INTERNAL && errno == ERANGE)
			    {
			      if (!scratch_buffer_grow (&tmpbuf))
				goto done;
			    }
			  else
			    break;
			}

		      if (h && (c = strchr (h->h_name, '.')))
			domain = __strdup (++c);
		    }
		}
	    }
	done:
	  scratch_buffer_free (&tmpbuf);
	}

      __libc_lock_unlock (lock);
    }

  return domain;
};

/* Copy a string to a destination buffer with length checking.  Return
   EAI_OVERFLOW if the buffer is not large enough, and 0 on
   success.  */
static int
checked_copy (char *dest, size_t destlen, const char *source)
{
  size_t source_length = strlen (source);
  if (source_length + 1 > destlen)
    return EAI_OVERFLOW;
  memcpy (dest, source, source_length + 1);
  return 0;
}

/* Helper function for CHECKED_SNPRINTF below.  */
static int
check_sprintf_result (int result, size_t destlen)
{
  if (result < 0)
    return EAI_SYSTEM;
  if ((size_t) result >= destlen)
    /* If ret == destlen, there was no room for the terminating NUL
       character.  */
    return EAI_OVERFLOW;
  return 0;
}

/* Format a string in the destination buffer.  Return 0 on success,
   EAI_OVERFLOW in case the buffer is too small, or EAI_SYSTEM on any
   other error.  */
#define CHECKED_SNPRINTF(dest, destlen, format, ...)			\
  check_sprintf_result							\
    (__snprintf (dest, destlen, format, __VA_ARGS__), destlen)

/* Convert host name, AF_INET/AF_INET6 case, name only.  */
static int
gni_host_inet_name (struct scratch_buffer *tmpbuf,
		    const struct sockaddr *sa, socklen_t addrlen,
		    char *host, socklen_t hostlen, int flags)
{
  int herrno;
  struct hostent th;
  struct hostent *h = NULL;
  if (sa->sa_family == AF_INET6)
    {
      const struct sockaddr_in6 *sin6p = (const struct sockaddr_in6 *) sa;
      while (__gethostbyaddr_r (&sin6p->sin6_addr, sizeof(struct in6_addr),
				AF_INET6, &th, tmpbuf->data, tmpbuf->length,
				&h, &herrno))
	if (herrno == NETDB_INTERNAL && errno == ERANGE)
	  {
	    if (!scratch_buffer_grow (tmpbuf))
	      {
		__set_h_errno (herrno);
		return EAI_MEMORY;
	      }
	  }
	else
	  break;
    }
  else
    {
      const struct sockaddr_in *sinp = (const struct sockaddr_in *) sa;
      while (__gethostbyaddr_r (&sinp->sin_addr, sizeof(struct in_addr),
				AF_INET, &th, tmpbuf->data, tmpbuf->length,
				&h, &herrno))
	if (herrno == NETDB_INTERNAL && errno == ERANGE)
	    {
	      if (!scratch_buffer_grow (tmpbuf))
		{
		  __set_h_errno (herrno);
		  return EAI_MEMORY;
		}
	    }
	else
	  break;
    }

  if (h == NULL)
    {
      if (herrno == NETDB_INTERNAL)
	{
	  __set_h_errno (herrno);
	  return EAI_SYSTEM;
	}
      if (herrno == TRY_AGAIN)
	{
	  __set_h_errno (herrno);
	  return EAI_AGAIN;
	}
    }

  if (h)
    {
      char *c;
      if ((flags & NI_NOFQDN)
	  && (c = nrl_domainname ())
	  && (c = strstr (h->h_name, c))
	  && (c != h->h_name) && (*(--c) == '.'))
	/* Terminate the string after the prefix.  */
	*c = '\0';

      /* If requested, convert from the IDN format.  */
      bool do_idn = flags & NI_IDN;
      char *h_name;
      if (do_idn)
	{
	  int rc = __idna_from_dns_encoding (h->h_name, &h_name);
	  if (rc == EAI_IDN_ENCODE)
	    /* Use the punycode name as a fallback.  */
	    do_idn = false;
	  else if (rc != 0)
	    return rc;
	}
      if (!do_idn)
	h_name = h->h_name;

      size_t len = strlen (h_name) + 1;
      if (len > hostlen)
	return EAI_OVERFLOW;
      memcpy (host, h_name, len);

      if (do_idn)
	free (h_name);

      return 0;
    }

  return EAI_NONAME;
}

/* Convert host name, AF_INET/AF_INET6 case, numeric conversion.  */
static int
gni_host_inet_numeric (struct scratch_buffer *tmpbuf,
		       const struct sockaddr *sa, socklen_t addrlen,
		       char *host, socklen_t hostlen, int flags)
{
  if (sa->sa_family == AF_INET6)
    {
      const struct sockaddr_in6 *sin6p = (const struct sockaddr_in6 *) sa;
      if (inet_ntop (AF_INET6, &sin6p->sin6_addr, host, hostlen) == NULL)
	return EAI_OVERFLOW;

      uint32_t scopeid = sin6p->sin6_scope_id;
      if (scopeid != 0)
	{
	  size_t used_hostlen = __strnlen (host, hostlen);
	  /* Location of the scope string in the host buffer.  */
	  char *scope_start = host + used_hostlen;
	  size_t scope_length = hostlen - used_hostlen;

	  if (IN6_IS_ADDR_LINKLOCAL (&sin6p->sin6_addr)
	      || IN6_IS_ADDR_MC_LINKLOCAL (&sin6p->sin6_addr))
	    {
	      char scopebuf[IFNAMSIZ];
	      if (if_indextoname (scopeid, scopebuf) != NULL)
		return CHECKED_SNPRINTF
		  (scope_start, scope_length,
		   "%c%s", SCOPE_DELIMITER, scopebuf);
	    }
	  return CHECKED_SNPRINTF
	    (scope_start, scope_length, "%c%u", SCOPE_DELIMITER, scopeid);
	}
    }
  else
    {
      const struct sockaddr_in *sinp = (const struct sockaddr_in *) sa;
      if (inet_ntop (AF_INET, &sinp->sin_addr, host, hostlen) == NULL)
	return EAI_OVERFLOW;
    }
  return 0;
}

/* Convert AF_INET or AF_INET6 socket address, host part.  */
static int
gni_host_inet (struct scratch_buffer *tmpbuf,
	       const struct sockaddr *sa, socklen_t addrlen,
	       char *host, socklen_t hostlen, int flags)
{
  if (!(flags & NI_NUMERICHOST))
    {
      int result = gni_host_inet_name
	(tmpbuf, sa, addrlen, host, hostlen, flags);
      if (result != EAI_NONAME)
	return result;
    }

  if (flags & NI_NAMEREQD)
    return EAI_NONAME;
  else
    return gni_host_inet_numeric
      (tmpbuf, sa, addrlen, host, hostlen, flags);
}

/* Convert AF_LOCAL socket address, host part.   */
static int
gni_host_local (struct scratch_buffer *tmpbuf,
		const struct sockaddr *sa, socklen_t addrlen,
		char *host, socklen_t hostlen, int flags)
{
  if (!(flags & NI_NUMERICHOST))
    {
      struct utsname utsname;
      if (uname (&utsname) == 0)
	return checked_copy (host, hostlen, utsname.nodename);
    }

  if (flags & NI_NAMEREQD)
    return EAI_NONAME;

  return checked_copy (host, hostlen, "localhost");
}

/* Convert the host part of an AF_LOCAK socket address.   */
static int
gni_host (struct scratch_buffer *tmpbuf,
	  const struct sockaddr *sa, socklen_t addrlen,
	  char *host, socklen_t hostlen, int flags)
{
  switch (sa->sa_family)
    {
    case AF_INET:
    case AF_INET6:
      return gni_host_inet (tmpbuf, sa, addrlen, host, hostlen, flags);

    case AF_LOCAL:
      return gni_host_local (tmpbuf, sa, addrlen, host, hostlen, flags);

    default:
      return EAI_FAMILY;
    }
}

/* Convert service to string, AF_INET and AF_INET6 variant.  */
static int
gni_serv_inet (struct scratch_buffer *tmpbuf,
	       const struct sockaddr *sa, socklen_t addrlen,
	       char *serv, socklen_t servlen, int flags)
{
  _Static_assert
    (offsetof (struct sockaddr_in, sin_port)
     == offsetof (struct sockaddr_in6, sin6_port)
     && sizeof (((struct sockaddr_in) {}).sin_port) == sizeof (in_port_t)
     && sizeof (((struct sockaddr_in6) {}).sin6_port) == sizeof (in_port_t),
     "AF_INET and AF_INET6 port consistency");
  const struct sockaddr_in *sinp = (const struct sockaddr_in *) sa;
  if (!(flags & NI_NUMERICSERV))
    {
      struct servent *s, ts;
      int e;
      while ((e = __getservbyport_r (sinp->sin_port,
				     ((flags & NI_DGRAM)
				      ? "udp" : "tcp"), &ts,
				     tmpbuf->data, tmpbuf->length, &s)))
	{
	  if (e == ERANGE)
	    {
	      if (!scratch_buffer_grow (tmpbuf))
		return EAI_MEMORY;
	    }
	  else
	    break;
	}
      if (s)
	return checked_copy (serv, servlen, s->s_name);
      /* Fall through to numeric conversion.  */
    }
  return CHECKED_SNPRINTF (serv, servlen, "%d", ntohs (sinp->sin_port));
}

/* Convert service to string, AF_LOCAL variant.  */
static int
gni_serv_local (struct scratch_buffer *tmpbuf,
	       const struct sockaddr *sa, socklen_t addrlen,
	       char *serv, socklen_t servlen, int flags)
{
  return checked_copy
    (serv, servlen, ((const struct sockaddr_un *) sa)->sun_path);
}

/* Convert service to string, dispatching to the implementations
   above.  */
static int
gni_serv (struct scratch_buffer *tmpbuf,
	  const struct sockaddr *sa, socklen_t addrlen,
	  char *serv, socklen_t servlen, int flags)
{
  switch (sa->sa_family)
    {
    case AF_INET:
    case AF_INET6:
      return gni_serv_inet (tmpbuf, sa, addrlen, serv, servlen, flags);
    case AF_LOCAL:
      return gni_serv_local (tmpbuf, sa, addrlen, serv, servlen, flags);
    default:
      return EAI_FAMILY;
    }
}

int
getnameinfo (const struct sockaddr *sa, socklen_t addrlen, char *host,
	     socklen_t hostlen, char *serv, socklen_t servlen,
	     int flags)
{
  if (flags & ~(NI_NUMERICHOST|NI_NUMERICSERV|NI_NOFQDN|NI_NAMEREQD|NI_DGRAM
		|NI_IDN|DEPRECATED_NI_IDN))
    return EAI_BADFLAGS;

  if (sa == NULL || addrlen < sizeof (sa_family_t))
    return EAI_FAMILY;

  if ((flags & NI_NAMEREQD) && host == NULL && serv == NULL)
    return EAI_NONAME;

  switch (sa->sa_family)
    {
    case AF_LOCAL:
      if (addrlen < (socklen_t) offsetof (struct sockaddr_un, sun_path))
	return EAI_FAMILY;
      break;
    case AF_INET:
      if (addrlen < sizeof (struct sockaddr_in))
	return EAI_FAMILY;
      break;
    case AF_INET6:
      if (addrlen < sizeof (struct sockaddr_in6))
	return EAI_FAMILY;
      break;
    default:
      return EAI_FAMILY;
    }

  struct scratch_buffer tmpbuf;
  scratch_buffer_init (&tmpbuf);

  if (host != NULL && hostlen > 0)
    {
      int result = gni_host (&tmpbuf, sa, addrlen, host, hostlen, flags);
      if (result != 0)
	{
	  scratch_buffer_free (&tmpbuf);
	  return result;
	}
    }

  if (serv && (servlen > 0))
    {
      int result = gni_serv (&tmpbuf, sa, addrlen, serv, servlen, flags);
      if (result != 0)
	{
	  scratch_buffer_free (&tmpbuf);
	  return result;
	}
    }

  scratch_buffer_free (&tmpbuf);
  return 0;
}
libc_hidden_def (getnameinfo)
