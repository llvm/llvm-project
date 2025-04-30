/*
 * Copyright (c) 2004 by Internet Systems Consortium, Inc. ("ISC")
 * Copyright (c) 1996-1999 by Internet Software Consortium.
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND INTERNET SOFTWARE CONSORTIUM DISCLAIMS
 * ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL INTERNET SOFTWARE
 * CONSORTIUM BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL
 * DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR
 * PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS
 * SOFTWARE.
 */

/* Import. */

#include <sys/types.h>
#include <sys/socket.h>

#include <netinet/in.h>
#include <arpa/nameser.h>
#include <arpa/inet.h>

#include <assert.h>
#include <errno.h>
#include <resolv.h>
#include <string.h>
#include <ctype.h>

#define SPRINTF(x) ((size_t)sprintf x)

/* Forward. */

static size_t	prune_origin(const char *name, const char *origin);
static int	charstr(const u_char *rdata, const u_char *edata,
			char **buf, size_t *buflen);
static int	addname(const u_char *msg, size_t msglen,
			const u_char **p, const char *origin,
			char **buf, size_t *buflen);
static void	addlen(size_t len, char **buf, size_t *buflen);
static int	addstr(const char *src, size_t len,
		       char **buf, size_t *buflen);
static int	addtab(size_t len, size_t target, int spaced,
		       char **buf, size_t *buflen);

/* Macros. */

#define	T(x) \
	do { \
		if ((x) < 0) \
			return (-1); \
	} while (0)

/* Public. */

/*%
 *	Convert an RR to presentation format.
 *
 * return:
 *\li	Number of characters written to buf, or -1 (check errno).
 */
int
ns_sprintrr(const ns_msg *handle, const ns_rr *rr,
	    const char *name_ctx, const char *origin,
	    char *buf, size_t buflen)
{
	int n;

	n = ns_sprintrrf(ns_msg_base(*handle), ns_msg_size(*handle),
			 ns_rr_name(*rr), ns_rr_class(*rr), ns_rr_type(*rr),
			 ns_rr_ttl(*rr), ns_rr_rdata(*rr), ns_rr_rdlen(*rr),
			 name_ctx, origin, buf, buflen);
	return (n);
}
libresolv_hidden_def (ns_sprintrr)

/*%
 *	Convert the fields of an RR into presentation format.
 *
 * return:
 *\li	Number of characters written to buf, or -1 (check errno).
 */
int
ns_sprintrrf(const u_char *msg, size_t msglen,
	    const char *name, ns_class class, ns_type type,
	    u_long ttl, const u_char *rdata, size_t rdlen,
	    const char *name_ctx, const char *origin,
	    char *buf, size_t buflen)
{
	const char *obuf = buf;
	const u_char *edata = rdata + rdlen;
	int spaced = 0;

	const char *comment;
	char tmp[100];
	char errbuf[40];
	int len, x;

	/*
	 * Owner.
	 */
	if (name_ctx != NULL && __libc_ns_samename (name_ctx, name) == 1)
		T(addstr("\t\t\t", 3, &buf, &buflen));
	else {
		len = prune_origin(name, origin);
		if (*name == '\0') {
			goto root;
		} else if (len == 0) {
			T(addstr("@\t\t\t", 4, &buf, &buflen));
		} else {
			T(addstr(name, len, &buf, &buflen));
			/* Origin not used or not root, and no trailing dot? */
			if (((origin == NULL || origin[0] == '\0') ||
			     (origin[0] != '.' && origin[1] != '\0' &&
			      name[len] == '\0')) && name[len - 1] != '.') {
 root:
				T(addstr(".", 1, &buf, &buflen));
				len++;
			}
			T(spaced = addtab(len, 24, spaced, &buf, &buflen));
		}
	}

	/*
	 * TTL, Class, Type.
	 */
	T(x = ns_format_ttl(ttl, buf, buflen));
	addlen(x, &buf, &buflen);
	len = SPRINTF((tmp, " %s %s", p_class(class), p_type(type)));
	T(addstr(tmp, len, &buf, &buflen));
	T(spaced = addtab(x + len, 16, spaced, &buf, &buflen));

	/*
	 * RData.
	 */
	switch (type) {
	case ns_t_a:
	  if (rdlen != (size_t)NS_INADDRSZ)
			goto formerr;
		(void) inet_ntop(AF_INET, rdata, buf, buflen);
		addlen(strlen(buf), &buf, &buflen);
		break;

	case ns_t_cname:
	case ns_t_mb:
	case ns_t_mg:
	case ns_t_mr:
	case ns_t_ns:
	case ns_t_ptr:
	case ns_t_dname:
		T(addname(msg, msglen, &rdata, origin, &buf, &buflen));
		break;

	case ns_t_hinfo:
	case ns_t_isdn:
		/* First word. */
		T(len = charstr(rdata, edata, &buf, &buflen));
		if (len == 0)
			goto formerr;
		rdata += len;
		T(addstr(" ", 1, &buf, &buflen));


		/* Second word, optional in ISDN records. */
		if (type == ns_t_isdn && rdata == edata)
			break;

		T(len = charstr(rdata, edata, &buf, &buflen));
		if (len == 0)
			goto formerr;
		rdata += len;
		break;

	case ns_t_soa: {
		u_long t;

		/* Server name. */
		T(addname(msg, msglen, &rdata, origin, &buf, &buflen));
		T(addstr(" ", 1, &buf, &buflen));

		/* Administrator name. */
		T(addname(msg, msglen, &rdata, origin, &buf, &buflen));
		T(addstr(" (\n", 3, &buf, &buflen));
		spaced = 0;

		if ((edata - rdata) != 5*NS_INT32SZ)
			goto formerr;

		/* Serial number. */
		t = ns_get32(rdata);  rdata += NS_INT32SZ;
		T(addstr("\t\t\t\t\t", 5, &buf, &buflen));
		len = SPRINTF((tmp, "%lu", t));
		T(addstr(tmp, len, &buf, &buflen));
		T(spaced = addtab(len, 16, spaced, &buf, &buflen));
		T(addstr("; serial\n", 9, &buf, &buflen));
		spaced = 0;

		/* Refresh interval. */
		t = ns_get32(rdata);  rdata += NS_INT32SZ;
		T(addstr("\t\t\t\t\t", 5, &buf, &buflen));
		T(len = ns_format_ttl(t, buf, buflen));
		addlen(len, &buf, &buflen);
		T(spaced = addtab(len, 16, spaced, &buf, &buflen));
		T(addstr("; refresh\n", 10, &buf, &buflen));
		spaced = 0;

		/* Retry interval. */
		t = ns_get32(rdata);  rdata += NS_INT32SZ;
		T(addstr("\t\t\t\t\t", 5, &buf, &buflen));
		T(len = ns_format_ttl(t, buf, buflen));
		addlen(len, &buf, &buflen);
		T(spaced = addtab(len, 16, spaced, &buf, &buflen));
		T(addstr("; retry\n", 8, &buf, &buflen));
		spaced = 0;

		/* Expiry. */
		t = ns_get32(rdata);  rdata += NS_INT32SZ;
		T(addstr("\t\t\t\t\t", 5, &buf, &buflen));
		T(len = ns_format_ttl(t, buf, buflen));
		addlen(len, &buf, &buflen);
		T(spaced = addtab(len, 16, spaced, &buf, &buflen));
		T(addstr("; expiry\n", 9, &buf, &buflen));
		spaced = 0;

		/* Minimum TTL. */
		t = ns_get32(rdata);  rdata += NS_INT32SZ;
		T(addstr("\t\t\t\t\t", 5, &buf, &buflen));
		T(len = ns_format_ttl(t, buf, buflen));
		addlen(len, &buf, &buflen);
		T(addstr(" )", 2, &buf, &buflen));
		T(spaced = addtab(len, 16, spaced, &buf, &buflen));
		T(addstr("; minimum\n", 10, &buf, &buflen));

		break;
	    }

	case ns_t_mx:
	case ns_t_afsdb:
	case ns_t_rt: {
		u_int t;

		if (rdlen < (size_t)NS_INT16SZ)
			goto formerr;

		/* Priority. */
		t = ns_get16(rdata);
		rdata += NS_INT16SZ;
		len = SPRINTF((tmp, "%u ", t));
		T(addstr(tmp, len, &buf, &buflen));

		/* Target. */
		T(addname(msg, msglen, &rdata, origin, &buf, &buflen));

		break;
	    }

	case ns_t_px: {
		u_int t;

		if (rdlen < (size_t)NS_INT16SZ)
			goto formerr;

		/* Priority. */
		t = ns_get16(rdata);
		rdata += NS_INT16SZ;
		len = SPRINTF((tmp, "%u ", t));
		T(addstr(tmp, len, &buf, &buflen));

		/* Name1. */
		T(addname(msg, msglen, &rdata, origin, &buf, &buflen));
		T(addstr(" ", 1, &buf, &buflen));

		/* Name2. */
		T(addname(msg, msglen, &rdata, origin, &buf, &buflen));

		break;
	    }

	case ns_t_x25:
		T(len = charstr(rdata, edata, &buf, &buflen));
		if (len == 0)
			goto formerr;
		rdata += len;
		break;

	case ns_t_txt:
		while (rdata < edata) {
			T(len = charstr(rdata, edata, &buf, &buflen));
			if (len == 0)
				goto formerr;
			rdata += len;
			if (rdata < edata)
				T(addstr(" ", 1, &buf, &buflen));
		}
		break;

	case ns_t_nsap: {
		char t[2+255*3];

		(void) inet_nsap_ntoa(rdlen, rdata, t);
		T(addstr(t, strlen(t), &buf, &buflen));
		break;
	    }

	case ns_t_aaaa:
	  if (rdlen != (size_t)NS_IN6ADDRSZ)
			goto formerr;
		(void) inet_ntop(AF_INET6, rdata, buf, buflen);
		addlen(strlen(buf), &buf, &buflen);
		break;

	case ns_t_loc: {
		char t[255];

		/* XXX protocol format checking? */
		(void) loc_ntoa(rdata, t);
		T(addstr(t, strlen(t), &buf, &buflen));
		break;
	    }

	case ns_t_naptr: {
		u_int order, preference;
		char t[50];

		if (rdlen < 2U*NS_INT16SZ)
			goto formerr;

		/* Order, Precedence. */
		order = ns_get16(rdata);	rdata += NS_INT16SZ;
		preference = ns_get16(rdata);	rdata += NS_INT16SZ;
		len = SPRINTF((t, "%u %u ", order, preference));
		T(addstr(t, len, &buf, &buflen));

		/* Flags. */
		T(len = charstr(rdata, edata, &buf, &buflen));
		if (len == 0)
			goto formerr;
		rdata += len;
		T(addstr(" ", 1, &buf, &buflen));

		/* Service. */
		T(len = charstr(rdata, edata, &buf, &buflen));
		if (len == 0)
			goto formerr;
		rdata += len;
		T(addstr(" ", 1, &buf, &buflen));

		/* Regexp. */
		T(len = charstr(rdata, edata, &buf, &buflen));
		if (len < 0)
			return (-1);
		if (len == 0)
			goto formerr;
		rdata += len;
		T(addstr(" ", 1, &buf, &buflen));

		/* Server. */
		T(addname(msg, msglen, &rdata, origin, &buf, &buflen));
		break;
	    }

	case ns_t_srv: {
		u_int priority, weight, port;
		char t[50];

		if (rdlen < 3U*NS_INT16SZ)
			goto formerr;

		/* Priority, Weight, Port. */
		priority = ns_get16(rdata);  rdata += NS_INT16SZ;
		weight   = ns_get16(rdata);  rdata += NS_INT16SZ;
		port     = ns_get16(rdata);  rdata += NS_INT16SZ;
		len = SPRINTF((t, "%u %u %u ", priority, weight, port));
		T(addstr(t, len, &buf, &buflen));

		/* Server. */
		T(addname(msg, msglen, &rdata, origin, &buf, &buflen));
		break;
	    }

	case ns_t_minfo:
	case ns_t_rp:
		/* Name1. */
		T(addname(msg, msglen, &rdata, origin, &buf, &buflen));
		T(addstr(" ", 1, &buf, &buflen));

		/* Name2. */
		T(addname(msg, msglen, &rdata, origin, &buf, &buflen));

		break;

	case ns_t_wks: {
		int n, lcnt;

		if (rdlen < 1U + NS_INT32SZ)
			goto formerr;

		/* Address. */
		(void) inet_ntop(AF_INET, rdata, buf, buflen);
		addlen(strlen(buf), &buf, &buflen);
		rdata += NS_INADDRSZ;

		/* Protocol. */
		len = SPRINTF((tmp, " %u ( ", *rdata));
		T(addstr(tmp, len, &buf, &buflen));
		rdata += NS_INT8SZ;

		/* Bit map. */
		n = 0;
		lcnt = 0;
		while (rdata < edata) {
			u_int c = *rdata++;
			do {
				if (c & 0200) {
					if (lcnt == 0) {
						T(addstr("\n\t\t\t\t", 5,
							 &buf, &buflen));
						lcnt = 10;
						spaced = 0;
					}
					len = SPRINTF((tmp, "%d ", n));
					T(addstr(tmp, len, &buf, &buflen));
					lcnt--;
				}
				c <<= 1;
			} while (++n & 07);
		}
		T(addstr(")", 1, &buf, &buflen));

		break;
	    }

	case ns_t_cert: {
		u_int c_type, key_tag, alg;
		int n;
		unsigned int siz;
		char base64_cert[8192], tmp[40];
		const char *leader;

		c_type  = ns_get16(rdata); rdata += NS_INT16SZ;
		key_tag = ns_get16(rdata); rdata += NS_INT16SZ;
		alg = (u_int) *rdata++;

		len = SPRINTF((tmp, "%d %d %d ", c_type, key_tag, alg));
		T(addstr(tmp, len, &buf, &buflen));
		siz = (edata-rdata)*4/3 + 4; /* "+4" accounts for trailing \0 */
		if (siz > sizeof(base64_cert) * 3/4) {
			const char *str = "record too long to print";
			T(addstr(str, strlen(str), &buf, &buflen));
		}
		else {
			len = b64_ntop(rdata, edata-rdata, base64_cert, siz);

			if (len < 0)
				goto formerr;
			else if (len > 15) {
				T(addstr(" (", 2, &buf, &buflen));
				leader = "\n\t\t";
				spaced = 0;
			}
			else
				leader = " ";

			for (n = 0; n < len; n += 48) {
				T(addstr(leader, strlen(leader),
					 &buf, &buflen));
				T(addstr(base64_cert + n, MIN(len - n, 48),
					 &buf, &buflen));
			}
			if (len > 15)
				T(addstr(" )", 2, &buf, &buflen));
		}
		break;
	    }

	case ns_t_tkey: {
		/* KJD - need to complete this */
		u_long t;
		int mode, err, keysize;

		/* Algorithm name. */
		T(addname(msg, msglen, &rdata, origin, &buf, &buflen));
		T(addstr(" ", 1, &buf, &buflen));

		/* Inception. */
		t = ns_get32(rdata);  rdata += NS_INT32SZ;
		len = SPRINTF((tmp, "%lu ", t));
		T(addstr(tmp, len, &buf, &buflen));

		/* Experation. */
		t = ns_get32(rdata);  rdata += NS_INT32SZ;
		len = SPRINTF((tmp, "%lu ", t));
		T(addstr(tmp, len, &buf, &buflen));

		/* Mode , Error, Key Size. */
		/* Priority, Weight, Port. */
		mode = ns_get16(rdata);  rdata += NS_INT16SZ;
		err  = ns_get16(rdata);  rdata += NS_INT16SZ;
		keysize  = ns_get16(rdata);  rdata += NS_INT16SZ;
		len = SPRINTF((tmp, "%u %u %u ", mode, err, keysize));
		T(addstr(tmp, len, &buf, &buflen));

		/* XXX need to dump key, print otherdata length & other data */
		break;
	    }

	case ns_t_tsig: {
		/* BEW - need to complete this */
		int n;

		T(len = addname(msg, msglen, &rdata, origin, &buf, &buflen));
		T(addstr(" ", 1, &buf, &buflen));
		rdata += 8; /*%< time */
		n = ns_get16(rdata); rdata += INT16SZ;
		rdata += n; /*%< sig */
		n = ns_get16(rdata); rdata += INT16SZ; /*%< original id */
		sprintf(buf, "%d", ns_get16(rdata));
		rdata += INT16SZ;
		addlen(strlen(buf), &buf, &buflen);
		break;
	    }

	case ns_t_a6: {
		struct in6_addr a;
		int pbyte, pbit;

		/* prefix length */
		if (rdlen == 0U) goto formerr;
		len = SPRINTF((tmp, "%d ", *rdata));
		T(addstr(tmp, len, &buf, &buflen));
		pbit = *rdata;
		if (pbit > 128) goto formerr;
		pbyte = (pbit & ~7) / 8;
		rdata++;

		/* address suffix: provided only when prefix len != 128 */
		if (pbit < 128) {
			if (rdata + pbyte >= edata) goto formerr;
			memset(&a, 0, sizeof(a));
			memcpy(&a.s6_addr[pbyte], rdata, sizeof(a) - pbyte);
			(void) inet_ntop(AF_INET6, &a, buf, buflen);
			addlen(strlen(buf), &buf, &buflen);
			rdata += sizeof(a) - pbyte;
		}

		/* prefix name: provided only when prefix len > 0 */
		if (pbit == 0)
			break;
		if (rdata >= edata) goto formerr;
		T(addstr(" ", 1, &buf, &buflen));
		T(addname(msg, msglen, &rdata, origin, &buf, &buflen));

		break;
	    }

	case ns_t_opt: {
		len = SPRINTF((tmp, "%u bytes", class));
		T(addstr(tmp, len, &buf, &buflen));
		break;
	    }

	default:
		snprintf (errbuf, sizeof (errbuf), "unknown RR type %d", type);
		comment = errbuf;
		goto hexify;
	}
	return (buf - obuf);
 formerr:
	comment = "RR format error";
 hexify: {
	int n, m;
	char *p;

	len = SPRINTF((tmp, "\\# %u%s\t; %s", (unsigned)(edata - rdata),
		       rdlen != 0U ? " (" : "", comment));
	T(addstr(tmp, len, &buf, &buflen));
	while (rdata < edata) {
		p = tmp;
		p += SPRINTF((p, "\n\t"));
		spaced = 0;
		n = MIN(16, edata - rdata);
		for (m = 0; m < n; m++)
			p += SPRINTF((p, "%02x ", rdata[m]));
		T(addstr(tmp, p - tmp, &buf, &buflen));
		if (n < 16) {
			T(addstr(")", 1, &buf, &buflen));
			T(addtab(p - tmp + 1, 48, spaced, &buf, &buflen));
		}
		p = tmp;
		p += SPRINTF((p, "; "));
		for (m = 0; m < n; m++)
			*p++ = (isascii(rdata[m]) && isprint(rdata[m]))
				? rdata[m]
				: '.';
		T(addstr(tmp, p - tmp, &buf, &buflen));
		rdata += n;
	}
	return (buf - obuf);
    }
}
libresolv_hidden_def (ns_sprintrrf)

/* Private. */

/*%
 * size_t
 * prune_origin(name, origin)
 *	Find out if the name is at or under the current origin.
 * return:
 *	Number of characters in name before start of origin,
 *	or length of name if origin does not match.
 * notes:
 *	This function should share code with samedomain().
 */
static size_t
prune_origin(const char *name, const char *origin) {
	const char *oname = name;

	while (*name != '\0') {
		if (origin != NULL && __libc_ns_samename (name, origin) == 1)
			return (name - oname - (name > oname));
		while (*name != '\0') {
			if (*name == '\\') {
				name++;
				/* XXX need to handle \nnn form. */
				if (*name == '\0')
					break;
			} else if (*name == '.') {
				name++;
				break;
			}
			name++;
		}
	}
	return (name - oname);
}

/*%
 * int
 * charstr(rdata, edata, buf, buflen)
 *	Format a <character-string> into the presentation buffer.
 * return:
 *	Number of rdata octets consumed
 *	0 for protocol format error
 *	-1 for output buffer error
 * side effects:
 *	buffer is advanced on success.
 */
static int
charstr(const u_char *rdata, const u_char *edata, char **buf, size_t *buflen) {
	const u_char *odata = rdata;
	size_t save_buflen = *buflen;
	char *save_buf = *buf;

	if (addstr("\"", 1, buf, buflen) < 0)
		goto enospc;
	if (rdata < edata) {
		int n = *rdata;

		if (rdata + 1 + n <= edata) {
			rdata++;
			while (n-- > 0) {
				if (strchr("\n\"\\", *rdata) != NULL)
					if (addstr("\\", 1, buf, buflen) < 0)
						goto enospc;
				if (addstr((const char *)rdata, 1,
					   buf, buflen) < 0)
					goto enospc;
				rdata++;
			}
		}
	}
	if (addstr("\"", 1, buf, buflen) < 0)
		goto enospc;
	return (rdata - odata);
 enospc:
	__set_errno (ENOSPC);
	*buf = save_buf;
	*buflen = save_buflen;
	return (-1);
}

static int
addname(const u_char *msg, size_t msglen,
	const u_char **pp, const char *origin,
	char **buf, size_t *buflen)
{
	size_t newlen, save_buflen = *buflen;
	char *save_buf = *buf;
	int n;

	n = __libc_dn_expand (msg, msg + msglen, *pp, *buf, *buflen);
	if (n < 0)
		goto enospc;	/*%< Guess. */
	newlen = prune_origin(*buf, origin);
	if (**buf == '\0') {
		goto root;
	} else if (newlen == 0U) {
		/* Use "@" instead of name. */
		if (newlen + 2 > *buflen)
			goto enospc;        /* No room for "@\0". */
		(*buf)[newlen++] = '@';
		(*buf)[newlen] = '\0';
	} else {
		if (((origin == NULL || origin[0] == '\0') ||
		    (origin[0] != '.' && origin[1] != '\0' &&
		    (*buf)[newlen] == '\0')) && (*buf)[newlen - 1] != '.') {
			/* No trailing dot. */
 root:
			if (newlen + 2 > *buflen)
				goto enospc;	/* No room for ".\0". */
			(*buf)[newlen++] = '.';
			(*buf)[newlen] = '\0';
		}
	}
	*pp += n;
	addlen(newlen, buf, buflen);
	**buf = '\0';
	return (newlen);
 enospc:
	__set_errno (ENOSPC);
	*buf = save_buf;
	*buflen = save_buflen;
	return (-1);
}

static void
addlen(size_t len, char **buf, size_t *buflen) {
	assert(len <= *buflen);
	*buf += len;
	*buflen -= len;
}

static int
addstr(const char *src, size_t len, char **buf, size_t *buflen) {
	if (len >= *buflen) {
		__set_errno (ENOSPC);
		return (-1);
	}
	memcpy(*buf, src, len);
	addlen(len, buf, buflen);
	**buf = '\0';
	return (0);
}

static int
addtab(size_t len, size_t target, int spaced, char **buf, size_t *buflen) {
	size_t save_buflen = *buflen;
	char *save_buf = *buf;
	int t;

	if (spaced || len >= target - 1) {
		T(addstr("  ", 2, buf, buflen));
		spaced = 1;
	} else {
		for (t = (target - len - 1) / 8; t >= 0; t--)
			if (addstr("\t", 1, buf, buflen) < 0) {
				*buflen = save_buflen;
				*buf = save_buf;
				return (-1);
			}
		spaced = 0;
	}
	return (spaced);
}
