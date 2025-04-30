/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

/*
 * Copyright (c) 1996,1999 by Internet Software Consortium.
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

/*
 * This file is primarily maintained by <tytso@mit.edu> and <ghudson@mit.edu>.
 */

/*
 * hesiod.c --- the core portion of the hesiod resolver.
 *
 * This file is derived from the hesiod library from Project Athena;
 * It has been extensively rewritten by Theodore Ts'o to have a more
 * thread-safe interface.
 */

/* Imports */

#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/nameser.h>

#include <errno.h>
#include <netdb.h>
#include <resolv.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hesiod.h"
#include "hesiod_p.h"

#define _PATH_HESIOD_CONF "/etc/hesiod.conf"

/* Forward */

static int	parse_config_file(struct hesiod_p *ctx, const char *filename);
static char **	get_txt_records(struct hesiod_p *ctx, int class,
				const char *name);

/* Public */

/*
 * This function is called to initialize a hesiod_p.
 */
int
hesiod_init(void **context) {
	struct hesiod_p *ctx;
	const char *configname;
	char *cp;

	ctx = malloc(sizeof(struct hesiod_p));
	if (ctx == 0)
		return (-1);

	ctx->LHS = NULL;
	ctx->RHS = NULL;
	/* Set default query classes. */
	ctx->classes[0] = C_IN;
	ctx->classes[1] = C_HS;

	configname = __libc_secure_getenv("HESIOD_CONFIG");
	if (!configname)
	  configname = _PATH_HESIOD_CONF;
	if (parse_config_file(ctx, configname) < 0) {
		goto cleanup;
	}
	/*
	 * The default RHS can be overridden by an environment
	 * variable.
	 */
	if ((cp = __libc_secure_getenv("HES_DOMAIN")) != NULL) {
		free(ctx->RHS);
		ctx->RHS = malloc(strlen(cp)+2);
		if (!ctx->RHS)
			goto cleanup;
		if (cp[0] == '.')
			strcpy(ctx->RHS, cp);
		else {
			ctx->RHS[0] = '.';
			strcpy(ctx->RHS + 1, cp);
		}
	}

	/*
	 * If there is no default hesiod realm set, we return an
	 * error.
	 */
	if (!ctx->RHS) {
		__set_errno(ENOEXEC);
		goto cleanup;
	}

	*context = ctx;
	return (0);

 cleanup:
	hesiod_end(ctx);
	return (-1);
}

/*
 * This function deallocates the hesiod_p
 */
void
hesiod_end(void *context) {
	struct hesiod_p *ctx = (struct hesiod_p *) context;
	int save_errno = errno;

	free(ctx->RHS);
	free(ctx->LHS);
	free(ctx);
	__set_errno(save_errno);
}

/*
 * This function takes a hesiod (name, type) and returns a DNS
 * name which is to be resolved.
 */
char *
hesiod_to_bind(void *context, const char *name, const char *type) {
	struct hesiod_p *ctx = (struct hesiod_p *) context;
	char *bindname;
	char **rhs_list = NULL;
	const char *RHS, *cp;
	char *endp;

	/* Decide what our RHS is, and set cp to the end of the actual name. */
	if ((cp = strchr(name, '@')) != NULL) {
		if (strchr(cp + 1, '.'))
			RHS = cp + 1;
		else if ((rhs_list = hesiod_resolve(context, cp + 1,
		    "rhs-extension")) != NULL)
			RHS = *rhs_list;
		else {
			__set_errno(ENOENT);
			return (NULL);
		}
	} else {
		RHS = ctx->RHS;
		cp = name + strlen(name);
	}

	/*
	 * Allocate the space we need, including up to three periods and
	 * the terminating NUL.
	 */
	if ((bindname = malloc((cp - name) + strlen(type) + strlen(RHS) +
	    (ctx->LHS ? strlen(ctx->LHS) : 0) + 4)) == NULL) {
		if (rhs_list)
			hesiod_free_list(context, rhs_list);
		return NULL;
	}

	/* Now put together the DNS name. */
	endp = (char *) __mempcpy (bindname, name, cp - name);
	*endp++ = '.';
	endp = (char *) __stpcpy (endp, type);
	if (ctx->LHS) {
		if (ctx->LHS[0] != '.')
			*endp++ = '.';
		endp = __stpcpy (endp, ctx->LHS);
	}
	if (RHS[0] != '.')
		*endp++ = '.';
	strcpy (endp, RHS);

	if (rhs_list)
		hesiod_free_list(context, rhs_list);

	return (bindname);
}

/*
 * This is the core function.  Given a hesiod (name, type), it
 * returns an array of strings returned by the resolver.
 */
char **
hesiod_resolve(void *context, const char *name, const char *type) {
	struct hesiod_p *ctx = (struct hesiod_p *) context;
	char *bindname = hesiod_to_bind(context, name, type);
	char **retvec;

	if (bindname == NULL)
		return (NULL);

	retvec = get_txt_records(ctx, ctx->classes[0], bindname);

	if (retvec == NULL && (errno == ENOENT || errno == ECONNREFUSED) && ctx->classes[1])
		retvec = get_txt_records(ctx, ctx->classes[1], bindname);


	free(bindname);
	return (retvec);
}

void
hesiod_free_list(void *context, char **list) {
	char **p;

	for (p = list; *p; p++)
		free(*p);
	free(list);
}

/*
 * This function parses the /etc/hesiod.conf file
 */
static int
parse_config_file(struct hesiod_p *ctx, const char *filename) {
	char buf[MAXDNAME+7];
	FILE *fp;

	/*
	 * Clear the existing configuration variable, just in case
	 * they're set.
	 */
	free(ctx->RHS);
	free(ctx->LHS);
	ctx->RHS = ctx->LHS = 0;
	/* Set default query classes. */
	ctx->classes[0] = C_IN;
	ctx->classes[1] = C_HS;

	/*
	 * Now open and parse the file...
	 */
	if (!(fp = fopen(filename, "rce")))
		return (-1);

	while (fgets(buf, sizeof(buf), fp) != NULL) {
		char *key, *data, *cp, **cpp;

		cp = buf;
		if (*cp == '#' || *cp == '\n' || *cp == '\r')
			continue;
		while(*cp == ' ' || *cp == '\t')
			cp++;
		key = cp;
		while(*cp != ' ' && *cp != '\t' && *cp != '=')
			cp++;
		*cp++ = '\0';

		while(*cp == ' ' || *cp == '\t' || *cp == '=')
			cp++;
		data = cp;
		while(*cp != ' ' && *cp != '\n' && *cp != '\r')
			cp++;
		*cp++ = '\0';

		cpp = NULL;
		if (strcasecmp(key, "lhs") == 0)
			cpp = &ctx->LHS;
		else if (strcasecmp(key, "rhs") == 0)
			cpp = &ctx->RHS;
		if (cpp) {
			*cpp = strdup(data);
			if (!*cpp)
				goto cleanup;
		} else if (strcasecmp(key, "classes") == 0) {
			int n = 0;
			while (*data && n < 2) {
				cp = strchrnul(data, ',');
				if (*cp != '\0')
					*cp++ = '\0';
				if (strcasecmp(data, "IN") == 0)
					ctx->classes[n++] = C_IN;
				else if (strcasecmp(data, "HS") == 0)
					ctx->classes[n++] = C_HS;
				data = cp;
			}
			if (n == 0) {
				/* Restore the default.  Better than
				   nother at all.  */
				ctx->classes[0] = C_IN;
				ctx->classes[1] = C_HS;
			} else if (n == 1
				   || ctx->classes[0] == ctx->classes[1])
				ctx->classes[1] = 0;
		}
	}
	fclose(fp);
	return (0);

 cleanup:
	fclose(fp);
	free(ctx->RHS);
	free(ctx->LHS);
	ctx->RHS = ctx->LHS = 0;
	return (-1);
}

/*
 * Given a DNS class and a DNS name, do a lookup for TXT records, and
 * return a list of them.
 */
static char **
get_txt_records(struct hesiod_p *ctx, int class, const char *name) {
	struct {
		int type;		/* RR type */
		int class;		/* RR class */
		int dlen;		/* len of data section */
		u_char *data;		/* pointer to data */
	} rr;
	HEADER *hp;
	u_char qbuf[MAX_HESRESP], abuf[MAX_HESRESP];
	u_char *cp, *erdata, *eom;
	char *dst, *edst, **list;
	int ancount, qdcount;
	int i, j, n, skip;

	/*
	 * Construct the query and send it.
	 */
	n = res_mkquery(QUERY, name, class, T_TXT, NULL, 0,
			 NULL, qbuf, MAX_HESRESP);
	if (n < 0) {
		__set_errno(EMSGSIZE);
		return (NULL);
	}
	n = res_send(qbuf, n, abuf, MAX_HESRESP);
	if (n < 0) {
		__set_errno(ECONNREFUSED);
		return (NULL);
	}
	if (n < HFIXEDSZ) {
		__set_errno(EMSGSIZE);
		return (NULL);
	}

	/*
	 * OK, parse the result.
	 */
	hp = (HEADER *) abuf;
	ancount = ntohs(hp->ancount);
	qdcount = ntohs(hp->qdcount);
	cp = abuf + sizeof(HEADER);
	eom = abuf + n;

	/* Skip query, trying to get to the answer section which follows. */
	for (i = 0; i < qdcount; i++) {
		skip = dn_skipname(cp, eom);
		if (skip < 0 || cp + skip + QFIXEDSZ > eom) {
			__set_errno(EMSGSIZE);
			return (NULL);
		}
		cp += skip + QFIXEDSZ;
	}

	list = malloc((ancount + 1) * sizeof(char *));
	if (!list)
		return (NULL);
	j = 0;
	for (i = 0; i < ancount; i++) {
		skip = dn_skipname(cp, eom);
		if (skip < 0) {
			__set_errno(EMSGSIZE);
			goto cleanup;
		}
		cp += skip;
		if (cp + 3 * INT16SZ + INT32SZ > eom) {
			__set_errno(EMSGSIZE);
			goto cleanup;
		}
		rr.type = ns_get16(cp);
		cp += INT16SZ;
		rr.class = ns_get16(cp);
		cp += INT16SZ + INT32SZ;	/* skip the ttl, too */
		rr.dlen = ns_get16(cp);
		cp += INT16SZ;
		if (rr.dlen == 0 || cp + rr.dlen > eom) {
			__set_errno(EMSGSIZE);
			goto cleanup;
		}
		rr.data = cp;
		cp += rr.dlen;
		if (rr.class != class || rr.type != T_TXT)
			continue;
		if (!(list[j] = malloc(rr.dlen)))
			goto cleanup;
		dst = list[j++];
		edst = dst + rr.dlen;
		erdata = rr.data + rr.dlen;
		cp = rr.data;
		while (cp < erdata) {
			n = (unsigned char) *cp++;
			if (cp + n > eom || dst + n > edst) {
				__set_errno(EMSGSIZE);
				goto cleanup;
			}
			memcpy(dst, cp, n);
			cp += n;
			dst += n;
		}
		if (cp != erdata) {
			__set_errno(EMSGSIZE);
			goto cleanup;
		}
		*dst = '\0';
	}
	list[j] = NULL;
	if (j == 0) {
		__set_errno(ENOENT);
		goto cleanup;
	}
	return (list);

 cleanup:
	for (i = 0; i < j; i++)
		free(list[i]);
	free(list);
	return (NULL);
}
