/*
 * ++Copyright++ 1985, 1988, 1993
 * -
 * Copyright (c) 1985, 1988, 1993
 *    The Regents of the University of California.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 4. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * -
 * Portions Copyright (c) 1993 by Digital Equipment Corporation.
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies, and that
 * the name of Digital Equipment Corporation not be used in advertising or
 * publicity pertaining to distribution of the document or software without
 * specific, written prior permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND DIGITAL EQUIPMENT CORP. DISCLAIMS ALL
 * WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS.   IN NO EVENT SHALL DIGITAL EQUIPMENT
 * CORPORATION BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL
 * DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR
 * PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS
 * SOFTWARE.
 * -
 * --Copyright--
 */

/* XXX This file is not used by any of the resolver functions implemented by
   glibc (i.e. get*info and gethostby*).  It cannot be removed however because
   it exports symbols in the libresolv ABI.  The file is not maintained any
   more, nor are these functions.  */

#include <shlib-compat.h>
#if SHLIB_COMPAT (libresolv, GLIBC_2_0, GLIBC_2_25)

# include <sys/types.h>
# include <sys/param.h>
# include <sys/socket.h>
# include <netinet/in.h>
# include <arpa/inet.h>
# include <arpa/nameser.h>

# include <stdio.h>
# include <netdb.h>
# include <resolv/resolv-internal.h>
# include <resolv/resolv_context.h>
# include <ctype.h>
# include <errno.h>
# include <stdlib.h>
# include <string.h>

# define	MAXALIASES	35
# define	MAXADDRS	35

static char *h_addr_ptrs[MAXADDRS + 1];

static struct hostent host;
static char *host_aliases[MAXALIASES];
static char hostbuf[8*1024];
static u_char host_addr[16];	/* IPv4 or IPv6 */
static FILE *hostf = NULL;
static int stayopen = 0;

static struct hostent *res_gethostbyname2_context (struct resolv_context *,
						   const char *name, int af);

static void map_v4v6_address (const char *src, char *dst) __THROW;
static void map_v4v6_hostent (struct hostent *hp, char **bp, int *len) __THROW;

extern void addrsort (char **, int) __THROW;

# if PACKETSZ > 65536
#  define	MAXPACKET	PACKETSZ
# else
#  define	MAXPACKET	65536
# endif

/* As per RFC 1034 and 1035 a host name cannot exceed 255 octets in length.  */
# ifdef MAXHOSTNAMELEN
#  undef MAXHOSTNAMELEN
# endif
# define MAXHOSTNAMELEN 256

typedef union {
    HEADER hdr;
    u_char buf[MAXPACKET];
} querybuf;

typedef union {
    int32_t al;
    char ac;
} align;

# ifndef h_errno
extern int h_errno;
# endif

# define BOUNDED_INCR(x) \
	do { \
		cp += x; \
		if (cp > eom) { \
			__set_h_errno (NO_RECOVERY); \
			return (NULL); \
		} \
	} while (0)

# define BOUNDS_CHECK(ptr, count) \
	do { \
		if ((ptr) + (count) > eom) { \
			__set_h_errno (NO_RECOVERY); \
			return (NULL); \
		} \
	} while (0)


static struct hostent *
getanswer (const querybuf *answer, int anslen, const char *qname, int qtype)
{
	const HEADER *hp;
	const u_char *cp;
	int n;
	const u_char *eom, *erdata;
	char *bp, **ap, **hap;
	int type, class, buflen, ancount, qdcount;
	int haveanswer, had_error;
	char tbuf[MAXDNAME];
	const char *tname;
	int (*name_ok) (const char *);

	tname = qname;
	host.h_name = NULL;
	eom = answer->buf + anslen;
	switch (qtype) {
	case T_A:
	case T_AAAA:
		name_ok = res_hnok;
		break;
	case T_PTR:
		name_ok = res_dnok;
		break;
	default:
		return (NULL);	/* XXX should be abort(); */
	}
	/*
	 * find first satisfactory answer
	 */
	hp = &answer->hdr;
	ancount = ntohs(hp->ancount);
	qdcount = ntohs(hp->qdcount);
	bp = hostbuf;
	buflen = sizeof hostbuf;
	cp = answer->buf;
	BOUNDED_INCR(HFIXEDSZ);
	if (qdcount != 1) {
		__set_h_errno (NO_RECOVERY);
		return (NULL);
	}
	n = __libc_dn_expand (answer->buf, eom, cp, bp, buflen);
	if ((n < 0) || !(*name_ok)(bp)) {
		__set_h_errno (NO_RECOVERY);
		return (NULL);
	}
	BOUNDED_INCR(n + QFIXEDSZ);
	if (qtype == T_A || qtype == T_AAAA) {
		/* res_send() has already verified that the query name is the
		 * same as the one we sent; this just gets the expanded name
		 * (i.e., with the succeeding search-domain tacked on).
		 */
		n = strlen(bp) + 1;		/* for the \0 */
		if (n >= MAXHOSTNAMELEN) {
			__set_h_errno (NO_RECOVERY);
			return (NULL);
		}
		host.h_name = bp;
		bp += n;
		buflen -= n;
		/* The qname can be abbreviated, but h_name is now absolute. */
		qname = host.h_name;
	}
	ap = host_aliases;
	*ap = NULL;
	host.h_aliases = host_aliases;
	hap = h_addr_ptrs;
	*hap = NULL;
	host.h_addr_list = h_addr_ptrs;
	haveanswer = 0;
	had_error = 0;
	while (ancount-- > 0 && cp < eom && !had_error) {
		n = __libc_dn_expand (answer->buf, eom, cp, bp, buflen);
		if ((n < 0) || !(*name_ok)(bp)) {
			had_error++;
			continue;
		}
		cp += n;			/* name */
		BOUNDS_CHECK(cp, 3 * INT16SZ + INT32SZ);
		type = ns_get16(cp);
		cp += INT16SZ;			/* type */
		class = ns_get16(cp);
		cp += INT16SZ + INT32SZ;	/* class, TTL */
		n = ns_get16(cp);
		cp += INT16SZ;			/* len */
		BOUNDS_CHECK(cp, n);
		erdata = cp + n;
		if (class != C_IN) {
			/* XXX - debug? syslog? */
			cp += n;
			continue;		/* XXX - had_error++ ? */
		}
		if ((qtype == T_A || qtype == T_AAAA) && type == T_CNAME) {
			if (ap >= &host_aliases[MAXALIASES-1])
				continue;
			n = __libc_dn_expand (answer->buf, eom, cp,
					      tbuf, sizeof tbuf);
			if ((n < 0) || !(*name_ok)(tbuf)) {
				had_error++;
				continue;
			}
			cp += n;
			if (cp != erdata) {
				__set_h_errno (NO_RECOVERY);
				return (NULL);
			}
			/* Store alias. */
			*ap++ = bp;
			n = strlen(bp) + 1;	/* for the \0 */
			if (n >= MAXHOSTNAMELEN) {
				had_error++;
				continue;
			}
			bp += n;
			buflen -= n;
			/* Get canonical name. */
			n = strlen(tbuf) + 1;	/* for the \0 */
			if (n > buflen || n >= MAXHOSTNAMELEN) {
				had_error++;
				continue;
			}
			strcpy(bp, tbuf);
			host.h_name = bp;
			bp += n;
			buflen -= n;
			continue;
		}
		if (qtype == T_PTR && type == T_CNAME) {
			n = __libc_dn_expand (answer->buf, eom, cp,
					      tbuf, sizeof tbuf);
			if (n < 0 || !res_dnok(tbuf)) {
				had_error++;
				continue;
			}
			cp += n;
			if (cp != erdata) {
				__set_h_errno (NO_RECOVERY);
				return (NULL);
			}
			/* Get canonical name. */
			n = strlen(tbuf) + 1;	/* for the \0 */
			if (n > buflen || n >= MAXHOSTNAMELEN) {
				had_error++;
				continue;
			}
			strcpy(bp, tbuf);
			tname = bp;
			bp += n;
			buflen -= n;
			continue;
		}
		if (type != qtype) {
			/* Log a low priority message if we get an unexpected
			 * record, but skip it if we are using DNSSEC since it
			 * uses many different types in responses that do not
			 * match QTYPE.
			 */
			cp += n;
			continue;		/* XXX - had_error++ ? */
		}
		switch (type) {
		case T_PTR:
			if (strcasecmp(tname, bp) != 0) {
				cp += n;
				continue;	/* XXX - had_error++ ? */
			}
			n = __libc_dn_expand (answer->buf, eom, cp,
					      bp, buflen);
			if ((n < 0) || !res_hnok(bp)) {
				had_error++;
				break;
			}
			cp += n;
			if (cp != erdata) {
				__set_h_errno (NO_RECOVERY);
				return (NULL);
			}
			if (!haveanswer)
				host.h_name = bp;
			else if (ap < &host_aliases[MAXALIASES-1])
				*ap++ = bp;
			else
				n = -1;
			if (n != -1) {
				n = strlen(bp) + 1;	/* for the \0 */
				if (n >= MAXHOSTNAMELEN) {
					had_error++;
					break;
				}
				bp += n;
				buflen -= n;
			}
			break;
		case T_A:
		case T_AAAA:
			if (strcasecmp(host.h_name, bp) != 0) {
				cp += n;
				continue;	/* XXX - had_error++ ? */
			}
			if (n != host.h_length) {
				cp += n;
				continue;
			}
			if (!haveanswer) {
				int nn;

				host.h_name = bp;
				nn = strlen(bp) + 1;	/* for the \0 */
				bp += nn;
				buflen -= nn;
			}

			/* XXX: when incrementing bp, we have to decrement
			 * buflen by the same amount --okir */
			buflen -= sizeof(align) - ((u_long)bp % sizeof(align));

			bp += sizeof(align) - ((u_long)bp % sizeof(align));

			if (bp + n >= &hostbuf[sizeof hostbuf]) {
				had_error++;
				continue;
			}
			if (hap >= &h_addr_ptrs[MAXADDRS-1]) {
				cp += n;
				continue;
			}
			memmove(*hap++ = bp, cp, n);
			bp += n;
			buflen -= n;
			cp += n;
			if (cp != erdata) {
				__set_h_errno (NO_RECOVERY);
				return (NULL);
			}
			break;
		default:
			abort();
		}
		if (!had_error)
			haveanswer++;
	}
	if (haveanswer) {
		*ap = NULL;
		*hap = NULL;
		/*
		 * Note: we sort even if host can take only one address
		 * in its return structures - should give it the "best"
		 * address in that case, not some random one
		 */
		if (_res.nsort && haveanswer > 1 && qtype == T_A)
			addrsort(h_addr_ptrs, haveanswer);
		if (!host.h_name) {
			n = strlen(qname) + 1;	/* for the \0 */
			if (n > buflen || n >= MAXHOSTNAMELEN)
				goto no_recovery;
			strcpy(bp, qname);
			host.h_name = bp;
			bp += n;
			buflen -= n;
		}
		if (res_use_inet6 ())
			map_v4v6_hostent(&host, &bp, &buflen);
		__set_h_errno (NETDB_SUCCESS);
		return (&host);
	}
 no_recovery:
	__set_h_errno (NO_RECOVERY);
	return (NULL);
}

extern struct hostent *res_gethostbyname2(const char *name, int af);
libresolv_hidden_proto (res_gethostbyname2)

struct hostent *
res_gethostbyname (const char *name)
{
  struct resolv_context *ctx = __resolv_context_get ();
  if (ctx == NULL)
    {
      __set_h_errno (NETDB_INTERNAL);
      return NULL;
    }

  if (res_use_inet6 ())
    {
      struct hostent *hp = res_gethostbyname2_context (ctx, name, AF_INET6);
      if (hp != NULL)
	{
	  __resolv_context_put (ctx);
	  return hp;
	}
    }
  struct hostent *hp = res_gethostbyname2_context (ctx, name, AF_INET);
  __resolv_context_put (ctx);
  return hp;
}
compat_symbol (libresolv, res_gethostbyname, res_gethostbyname, GLIBC_2_0);

static struct hostent *
res_gethostbyname2_context (struct resolv_context *ctx,
			    const char *name, int af)
{
	union
	{
	  querybuf *buf;
	  u_char *ptr;
	} buf;
	querybuf *origbuf;
	const char *cp;
	char *bp;
	int n, size, type, len;
	struct hostent *ret;

	switch (af) {
	case AF_INET:
		size = INADDRSZ;
		type = T_A;
		break;
	case AF_INET6:
		size = IN6ADDRSZ;
		type = T_AAAA;
		break;
	default:
		__set_h_errno (NETDB_INTERNAL);
		__set_errno (EAFNOSUPPORT);
		return (NULL);
	}

	host.h_addrtype = af;
	host.h_length = size;

	/*
	 * if there aren't any dots, it could be a user-level alias.
	 * this is also done in res_query() since we are not the only
	 * function that looks up host names.
	 */
	char abuf[MAXDNAME];
	if (strchr (name, '.') != NULL
	    && (cp = __res_context_hostalias (ctx, name, abuf, sizeof (abuf))))
	  name = cp;

	/*
	 * disallow names consisting only of digits/dots, unless
	 * they end in a dot.
	 */
	if (isdigit(name[0]))
		for (cp = name;; ++cp) {
			if (!*cp) {
				if (*--cp == '.')
					break;
				/*
				 * All-numeric, no dot at the end.
				 * Fake up a hostent as if we'd actually
				 * done a lookup.
				 */
				if (inet_pton(af, name, host_addr) <= 0) {
					__set_h_errno (HOST_NOT_FOUND);
					return (NULL);
				}
				strncpy(hostbuf, name, MAXDNAME);
				hostbuf[MAXDNAME] = '\0';
				bp = hostbuf + MAXDNAME;
				len = sizeof hostbuf - MAXDNAME;
				host.h_name = hostbuf;
				host.h_aliases = host_aliases;
				host_aliases[0] = NULL;
				h_addr_ptrs[0] = (char *)host_addr;
				h_addr_ptrs[1] = NULL;
				host.h_addr_list = h_addr_ptrs;
				if (res_use_inet6 ())
					map_v4v6_hostent(&host, &bp, &len);
				__set_h_errno (NETDB_SUCCESS);
				return (&host);
			}
			if (!isdigit(*cp) && *cp != '.')
				break;
	       }
	if ((isxdigit(name[0]) && strchr(name, ':') != NULL) ||
	    name[0] == ':')
		for (cp = name;; ++cp) {
			if (!*cp) {
				if (*--cp == '.')
					break;
				/*
				 * All-IPv6-legal, no dot at the end.
				 * Fake up a hostent as if we'd actually
				 * done a lookup.
				 */
				if (inet_pton(af, name, host_addr) <= 0) {
					__set_h_errno (HOST_NOT_FOUND);
					return (NULL);
				}
				strncpy(hostbuf, name, MAXDNAME);
				hostbuf[MAXDNAME] = '\0';
				bp = hostbuf + MAXDNAME;
				len = sizeof hostbuf - MAXDNAME;
				host.h_name = hostbuf;
				host.h_aliases = host_aliases;
				host_aliases[0] = NULL;
				h_addr_ptrs[0] = (char *)host_addr;
				h_addr_ptrs[1] = NULL;
				host.h_addr_list = h_addr_ptrs;
				__set_h_errno (NETDB_SUCCESS);
				return (&host);
			}
			if (!isxdigit(*cp) && *cp != ':' && *cp != '.')
				break;
		}

	buf.buf = origbuf = (querybuf *) alloca (1024);

	if ((n = __res_context_search
	     (ctx, name, C_IN, type, buf.buf->buf, 1024,
	      &buf.ptr, NULL, NULL, NULL, NULL)) < 0) {
		if (buf.buf != origbuf)
			free (buf.buf);
		if (errno == ECONNREFUSED)
			return (_gethtbyname2(name, af));
		return (NULL);
	}
	ret = getanswer(buf.buf, n, name, type);
	if (buf.buf != origbuf)
		free (buf.buf);
	return ret;
}

struct hostent *
res_gethostbyname2 (const char *name, int af)
{
  struct resolv_context *ctx = __resolv_context_get ();
  if (ctx == NULL)
    {
      __set_h_errno (NETDB_INTERNAL);
      return NULL;
    }
  struct hostent *hp = res_gethostbyname2_context (ctx, name, AF_INET);
  __resolv_context_put (ctx);
  return hp;
}
libresolv_hidden_def (res_gethostbyname2)
compat_symbol (libresolv, res_gethostbyname2, res_gethostbyname2, GLIBC_2_0);

static struct hostent *
res_gethostbyaddr_context (struct resolv_context *ctx,
			   const void *addr, socklen_t len, int af)
{
	const u_char *uaddr = (const u_char *)addr;
	static const u_char mapped[] = { 0,0, 0,0, 0,0, 0,0, 0,0, 0xff,0xff };
	static const u_char tunnelled[] = { 0,0, 0,0, 0,0, 0,0, 0,0, 0,0 };
	int n;
	socklen_t size;
	union
	{
	  querybuf *buf;
	  u_char *ptr;
	} buf;
	querybuf *orig_buf;
	struct hostent *hp;
	char qbuf[MAXDNAME+1], *qp = NULL;

	if (af == AF_INET6 && len == IN6ADDRSZ &&
	    (!memcmp(uaddr, mapped, sizeof mapped) ||
	     !memcmp(uaddr, tunnelled, sizeof tunnelled))) {
		/* Unmap. */
		addr += sizeof mapped;
		uaddr += sizeof mapped;
		af = AF_INET;
		len = INADDRSZ;
	}
	switch (af) {
	case AF_INET:
		size = INADDRSZ;
		break;
	case AF_INET6:
		size = IN6ADDRSZ;
		break;
	default:
		__set_errno (EAFNOSUPPORT);
		__set_h_errno (NETDB_INTERNAL);
		return (NULL);
	}
	if (size != len) {
		__set_errno (EINVAL);
		__set_h_errno (NETDB_INTERNAL);
		return (NULL);
	}
	switch (af) {
	case AF_INET:
		(void) sprintf(qbuf, "%u.%u.%u.%u.in-addr.arpa",
			       (uaddr[3] & 0xff),
			       (uaddr[2] & 0xff),
			       (uaddr[1] & 0xff),
			       (uaddr[0] & 0xff));
		break;
	case AF_INET6:
		qp = qbuf;
		for (n = IN6ADDRSZ - 1; n >= 0; n--) {
			qp += sprintf(qp, "%x.%x.",
				      uaddr[n] & 0xf,
				      (uaddr[n] >> 4) & 0xf);
		}
		strcpy(qp, "ip6.arpa");
		break;
	default:
		abort();
	}

	buf.buf = orig_buf = (querybuf *) alloca (1024);

	n = __res_context_query (ctx, qbuf, C_IN, T_PTR, buf.buf->buf, 1024,
				 &buf.ptr, NULL, NULL, NULL, NULL);
	if (n < 0) {
		if (buf.buf != orig_buf)
			free (buf.buf);
		if (errno == ECONNREFUSED)
			return (_gethtbyaddr(addr, len, af));
		return (NULL);
	}
	hp = getanswer(buf.buf, n, qbuf, T_PTR);
	if (buf.buf != orig_buf)
		free (buf.buf);
	if (!hp)
		return (NULL);	/* h_errno was set by getanswer() */
	hp->h_addrtype = af;
	hp->h_length = len;
	memmove(host_addr, addr, len);
	h_addr_ptrs[0] = (char *)host_addr;
	h_addr_ptrs[1] = NULL;
	if (af == AF_INET && res_use_inet6 ()) {
		map_v4v6_address((char*)host_addr, (char*)host_addr);
		hp->h_addrtype = AF_INET6;
		hp->h_length = IN6ADDRSZ;
	}
	__set_h_errno (NETDB_SUCCESS);
	return (hp);
}

struct hostent *
res_gethostbyaddr (const void *addr, socklen_t len, int af)
{
  struct resolv_context *ctx = __resolv_context_get ();
  if (ctx == NULL)
    {
      __set_h_errno (NETDB_INTERNAL);
      return NULL;
    }
  struct hostent *hp = res_gethostbyaddr_context (ctx, addr, len, af);
  __resolv_context_put (ctx);
  return hp;
}
compat_symbol (libresolv, res_gethostbyaddr, res_gethostbyaddr, GLIBC_2_0);

void
_sethtent (int f)
{
	if (!hostf)
		hostf = fopen(_PATH_HOSTS, "rce" );
	else
		rewind(hostf);
	stayopen = f;
}
libresolv_hidden_def (_sethtent)
compat_symbol (libresolv, _sethtent, _sethtent, GLIBC_2_0);

static void
_endhtent (void)
{
	if (hostf && !stayopen) {
		(void) fclose(hostf);
		hostf = NULL;
	}
}

struct hostent *
_gethtent (void)
{
	char *p;
	char *cp, **q;
	int af, len;

	if (!hostf && !(hostf = fopen(_PATH_HOSTS, "rce" ))) {
		__set_h_errno (NETDB_INTERNAL);
		return (NULL);
	}
 again:
	if (!(p = fgets(hostbuf, sizeof hostbuf, hostf))) {
		__set_h_errno (HOST_NOT_FOUND);
		return (NULL);
	}
	if (*p == '#')
		goto again;
	if (!(cp = strpbrk(p, "#\n")))
		goto again;
	*cp = '\0';
	if (!(cp = strpbrk(p, " \t")))
		goto again;
	*cp++ = '\0';
	if (inet_pton(AF_INET6, p, host_addr) > 0) {
		af = AF_INET6;
		len = IN6ADDRSZ;
	} else if (inet_pton(AF_INET, p, host_addr) > 0) {
		if (res_use_inet6 ()) {
			map_v4v6_address((char*)host_addr, (char*)host_addr);
			af = AF_INET6;
			len = IN6ADDRSZ;
		} else {
			af = AF_INET;
			len = INADDRSZ;
		}
	} else {
		goto again;
	}
	h_addr_ptrs[0] = (char *)host_addr;
	h_addr_ptrs[1] = NULL;
	host.h_addr_list = h_addr_ptrs;
	host.h_length = len;
	host.h_addrtype = af;
	while (*cp == ' ' || *cp == '\t')
		cp++;
	host.h_name = cp;
	q = host.h_aliases = host_aliases;
	if ((cp = strpbrk(cp, " \t")))
		*cp++ = '\0';
	while (cp && *cp) {
		if (*cp == ' ' || *cp == '\t') {
			cp++;
			continue;
		}
		if (q < &host_aliases[MAXALIASES - 1])
			*q++ = cp;
		if ((cp = strpbrk(cp, " \t")))
			*cp++ = '\0';
	}
	*q = NULL;
	__set_h_errno (NETDB_SUCCESS);
	return (&host);
}
libresolv_hidden_def (_gethtent)
compat_symbol (libresolv, _gethtent, _gethtent, GLIBC_2_0);

struct hostent *
_gethtbyname (const char *name)
{
	struct hostent *hp;

	if (res_use_inet6 ()) {
		hp = _gethtbyname2(name, AF_INET6);
		if (hp)
			return (hp);
	}
	return (_gethtbyname2(name, AF_INET));
}
compat_symbol (libresolv, _gethtbyname, _gethtbyname, GLIBC_2_0);

struct hostent *
_gethtbyname2 (const char *name, int af)
{
	struct hostent *p;
	char **cp;

	_sethtent(0);
	while ((p = _gethtent())) {
		if (p->h_addrtype != af)
			continue;
		if (strcasecmp(p->h_name, name) == 0)
			break;
		for (cp = p->h_aliases; *cp != 0; cp++)
			if (strcasecmp(*cp, name) == 0)
				goto found;
	}
 found:
	_endhtent();
	return (p);
}
libresolv_hidden_def (_gethtbyname2)
compat_symbol (libresolv, _gethtbyname2, _gethtbyname2, GLIBC_2_0);

struct hostent *
_gethtbyaddr (const char *addr, size_t len, int af)
{
	struct hostent *p;

	_sethtent(0);
	while ((p = _gethtent()))
		if (p->h_addrtype == af && !memcmp(p->h_addr, addr, len))
			break;
	_endhtent();
	return (p);
}
libresolv_hidden_def (_gethtbyaddr)
compat_symbol (libresolv, _gethtbyaddr, _gethtbyaddr, GLIBC_2_0);

static void
map_v4v6_address (const char *src, char *dst)
{
	u_char *p = (u_char *)dst;
	char tmp[INADDRSZ];
	int i;

	/* Stash a temporary copy so our caller can update in place. */
	memcpy(tmp, src, INADDRSZ);
	/* Mark this ipv6 addr as a mapped ipv4. */
	for (i = 0; i < 10; i++)
		*p++ = 0x00;
	*p++ = 0xff;
	*p++ = 0xff;
	/* Retrieve the saved copy and we're done. */
	memcpy((void*)p, tmp, INADDRSZ);
}

static void
map_v4v6_hostent (struct hostent *hp, char **bpp, int *lenp)
{
	char **ap;

	if (hp->h_addrtype != AF_INET || hp->h_length != INADDRSZ)
		return;
	hp->h_addrtype = AF_INET6;
	hp->h_length = IN6ADDRSZ;
	for (ap = hp->h_addr_list; *ap; ap++) {
		int i = sizeof(align) - ((u_long)*bpp % sizeof(align));

		if (*lenp < (i + IN6ADDRSZ)) {
			/* Out of memory.  Truncate address list here.  XXX */
			*ap = NULL;
			return;
		}
		*bpp += i;
		*lenp -= i;
		map_v4v6_address(*ap, *bpp);
		*ap = *bpp;
		*bpp += IN6ADDRSZ;
		*lenp -= IN6ADDRSZ;
	}
}

extern void
addrsort (char **ap, int num)
{
	int i, j;
	char **p;
	short aval[MAXADDRS];
	int needsort = 0;

	p = ap;
	for (i = 0; i < num; i++, p++) {
	    for (j = 0 ; (unsigned)j < _res.nsort; j++)
		if (_res.sort_list[j].addr.s_addr ==
		    (((struct in_addr *)(*p))->s_addr & _res.sort_list[j].mask))
			break;
	    aval[i] = j;
	    if (needsort == 0 && i > 0 && j < aval[i-1])
		needsort = i;
	}
	if (!needsort)
	    return;

	while (needsort < num) {
	    for (j = needsort - 1; j >= 0; j--) {
		if (aval[j] > aval[j+1]) {
		    char *hp;

		    i = aval[j];
		    aval[j] = aval[j+1];
		    aval[j+1] = i;

		    hp = ap[j];
		    ap[j] = ap[j+1];
		    ap[j+1] = hp;

		} else
		    break;
	    }
	    needsort++;
	}
}

#endif	/* SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_25) */
