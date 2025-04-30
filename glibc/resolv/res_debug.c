/*
 * Copyright (c) 1985
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
 */

/*
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
 */

/*
 * Portions Copyright (c) 1995 by International Business Machines, Inc.
 *
 * International Business Machines, Inc. (hereinafter called IBM) grants
 * permission under its copyrights to use, copy, modify, and distribute this
 * Software with or without fee, provided that the above copyright notice and
 * all paragraphs of this notice appear in all copies, and that the name of IBM
 * not be used in connection with the marketing of any product incorporating
 * the Software or modifications thereof, without specific, written prior
 * permission.
 *
 * To the extent it has a right to do so, IBM grants an immunity from suit
 * under its patents, if any, for the use, sale or manufacture of products to
 * the extent that such products are used for performing Domain Name System
 * dynamic updates in TCP/IP networks by means of the Software.  No immunity is
 * granted for any product per se or for any other function of any product.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", AND IBM DISCLAIMS ALL WARRANTIES,
 * INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL IBM BE LIABLE FOR ANY SPECIAL,
 * DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING
 * OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE, EVEN
 * IF IBM IS APPRISED OF THE POSSIBILITY OF SUCH DAMAGES.
 */

/*
 * Portions Copyright (c) 1996-1999 by Internet Software Consortium.
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

#include <sys/types.h>
#include <sys/param.h>
#include <sys/socket.h>

#include <netinet/in.h>
#include <arpa/inet.h>
#include <arpa/nameser.h>

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <netdb.h>
#include <resolv/resolv-internal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <shlib-compat.h>
#include <libc-diag.h>

#ifdef SPRINTF_CHAR
# define SPRINTF(x) strlen(sprintf/**/x)
#else
# define SPRINTF(x) sprintf x
#endif

extern const char *_res_sectioncodes[] attribute_hidden;

/* _res_opcodes was exported by accident as a variable.  */
#if SHLIB_COMPAT (libresolv, GLIBC_2_0, GLIBC_2_26)
static const char *res_opcodes[] =
#else
static const char res_opcodes[][9] =
#endif
  {
    "QUERY",
    "IQUERY",
    "CQUERYM",
    "CQUERYU",	/* experimental */
    "NOTIFY",	/* experimental */
    "UPDATE",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "ZONEINIT",
    "ZONEREF",
  };
#if SHLIB_COMPAT (libresolv, GLIBC_2_0, GLIBC_2_26)
strong_alias (res_opcodes, _res_opcodes)
#endif

static const char *p_section(int section, int opcode);

/*
 * Print the current options.
 */
void
fp_resstat(const res_state statp, FILE *file) {
	u_long mask;

	fprintf(file, ";; res options:");
	for (mask = 1;  mask != 0;  mask <<= 1)
		if (statp->options & mask)
			fprintf(file, " %s", p_option(mask));
	putc('\n', file);
}

static void
do_section (int pfcode, ns_msg *handle, ns_sect section, int pflag, FILE *file)
{
	int n, sflag, rrnum;
	static int buflen = 2048;
	char *buf;
	ns_opcode opcode;
	ns_rr rr;

	/*
	 * Print answer records.
	 */
	sflag = (pfcode & pflag);
	if (pfcode && !sflag)
		return;

	buf = malloc(buflen);
	if (buf == NULL) {
		fprintf(file, ";; memory allocation failure\n");
		return;
	}

	opcode = (ns_opcode) ns_msg_getflag(*handle, ns_f_opcode);
	rrnum = 0;
	for (;;) {
		if (ns_parserr(handle, section, rrnum, &rr)) {
			if (errno != ENODEV)
				fprintf(file, ";; ns_parserr: %s\n",
					strerror(errno));
			else if (rrnum > 0 && sflag != 0 &&
				 (pfcode & RES_PRF_HEAD1))
				putc('\n', file);
			goto cleanup;
		}
		if (rrnum == 0 && sflag != 0 && (pfcode & RES_PRF_HEAD1))
			fprintf(file, ";; %s SECTION:\n",
				p_section(section, opcode));
		if (section == ns_s_qd)
			fprintf(file, ";;\t%s, type = %s, class = %s\n",
				ns_rr_name(rr),
				p_type(ns_rr_type(rr)),
				p_class(ns_rr_class(rr)));
		else {
			n = ns_sprintrr(handle, &rr, NULL, NULL,
					buf, buflen);
			if (n < 0) {
				if (errno == ENOSPC) {
					free(buf);
					buf = NULL;
					if (buflen < 131072)
						buf = malloc(buflen += 1024);
					if (buf == NULL) {
						fprintf(file,
					      ";; memory allocation failure\n");
					      return;
					}
					continue;
				}
				fprintf(file, ";; ns_sprintrr: %s\n",
					strerror(errno));
				goto cleanup;
			}
			fputs(buf, file);
			fputc('\n', file);
		}
		rrnum++;
	}
 cleanup:
	free(buf);
}

/*
 * Print the contents of a query.
 * This is intended to be primarily a debugging routine.
 */
void
fp_nquery (const unsigned char *msg, int len, FILE *file)
{
	ns_msg handle;
	int qdcount, ancount, nscount, arcount;
	u_int opcode, rcode, id;

	/* There is no need to initialize _res: If _res is not yet
	   initialized, _res.pfcode is zero.  But initialization will
	   leave it at zero, too.  _res.pfcode is an unsigned long,
	   but the code here assumes that the flags fit into an int,
	   so use that.  */
	int pfcode = _res.pfcode;

	if (ns_initparse(msg, len, &handle) < 0) {
		fprintf(file, ";; ns_initparse: %s\n", strerror(errno));
		return;
	}
	opcode = ns_msg_getflag(handle, ns_f_opcode);
	rcode = ns_msg_getflag(handle, ns_f_rcode);
	id = ns_msg_id(handle);
	qdcount = ns_msg_count(handle, ns_s_qd);
	ancount = ns_msg_count(handle, ns_s_an);
	nscount = ns_msg_count(handle, ns_s_ns);
	arcount = ns_msg_count(handle, ns_s_ar);

	/*
	 * Print header fields.
	 */
	if ((!pfcode) || (pfcode & RES_PRF_HEADX) || rcode)
		fprintf(file,
			";; ->>HEADER<<- opcode: %s, status: %s, id: %d\n",
			res_opcodes[opcode], p_rcode(rcode), id);
	if ((!pfcode) || (pfcode & RES_PRF_HEADX))
		putc(';', file);
	if ((!pfcode) || (pfcode & RES_PRF_HEAD2)) {
		fprintf(file, "; flags:");
		if (ns_msg_getflag(handle, ns_f_qr))
			fprintf(file, " qr");
		if (ns_msg_getflag(handle, ns_f_aa))
			fprintf(file, " aa");
		if (ns_msg_getflag(handle, ns_f_tc))
			fprintf(file, " tc");
		if (ns_msg_getflag(handle, ns_f_rd))
			fprintf(file, " rd");
		if (ns_msg_getflag(handle, ns_f_ra))
			fprintf(file, " ra");
		if (ns_msg_getflag(handle, ns_f_z))
			fprintf(file, " ??");
		if (ns_msg_getflag(handle, ns_f_ad))
			fprintf(file, " ad");
		if (ns_msg_getflag(handle, ns_f_cd))
			fprintf(file, " cd");
	}
	if ((!pfcode) || (pfcode & RES_PRF_HEAD1)) {
		fprintf(file, "; %s: %d",
			p_section(ns_s_qd, opcode), qdcount);
		fprintf(file, ", %s: %d",
			p_section(ns_s_an, opcode), ancount);
		fprintf(file, ", %s: %d",
			p_section(ns_s_ns, opcode), nscount);
		fprintf(file, ", %s: %d",
			p_section(ns_s_ar, opcode), arcount);
	}
	if ((!pfcode) || (pfcode &
		(RES_PRF_HEADX | RES_PRF_HEAD2 | RES_PRF_HEAD1))) {
		putc('\n',file);
	}
	/*
	 * Print the various sections.
	 */
	do_section (pfcode, &handle, ns_s_qd, RES_PRF_QUES, file);
	do_section (pfcode, &handle, ns_s_an, RES_PRF_ANS, file);
	do_section (pfcode, &handle, ns_s_ns, RES_PRF_AUTH, file);
	do_section (pfcode, &handle, ns_s_ar, RES_PRF_ADD, file);
	if (qdcount == 0 && ancount == 0 &&
	    nscount == 0 && arcount == 0)
		putc('\n', file);
}
libresolv_hidden_def (fp_nquery)

void
fp_query (const unsigned char *msg, FILE *file)
{
  fp_nquery (msg, PACKETSZ, file);
}
libresolv_hidden_def (fp_query)

void
p_query (const unsigned char *msg)
{
  fp_query (msg, stdout);
}

const u_char *
p_cdnname(const u_char *cp, const u_char *msg, int len, FILE *file) {
	char name[MAXDNAME];
	int n;

	if ((n = __libc_dn_expand (msg, msg + len, cp, name, sizeof name)) < 0)
		return (NULL);
	if (name[0] == '\0')
		putc('.', file);
	else
		fputs(name, file);
	return (cp + n);
}
libresolv_hidden_def (p_cdnname)

const u_char *
p_cdname(const u_char *cp, const u_char *msg, FILE *file) {
	return (p_cdnname(cp, msg, PACKETSZ, file));
}

/* Return a fully-qualified domain name from a compressed name (with
   length supplied).  */

const u_char *
p_fqnname (const u_char *cp, const u_char *msg, int msglen, char *name,
	   int namelen)
{
	int n, newlen;

	if ((n = __libc_dn_expand (msg, cp + msglen, cp, name, namelen)) < 0)
		return (NULL);
	newlen = strlen(name);
	if (newlen == 0 || name[newlen - 1] != '.') {
		if (newlen + 1 >= namelen)	/* Lack space for final dot */
			return (NULL);
		else
			strcpy(name + newlen, ".");
	}
	return (cp + n);
}
libresolv_hidden_def (p_fqnname)

/* XXX:	the rest of these functions need to become length-limited, too. */

const u_char *
p_fqname(const u_char *cp, const u_char *msg, FILE *file) {
	char name[MAXDNAME];
	const u_char *n;

	n = p_fqnname(cp, msg, MAXCDNAME, name, sizeof name);
	if (n == NULL)
		return (NULL);
	fputs(name, file);
	return (n);
}

/*
 * Names of RR classes and qclasses.  Classes and qclasses are the same, except
 * that C_ANY is a qclass but not a class.  (You can ask for records of class
 * C_ANY, but you can't have any records of that class in the database.)
 */
extern const struct res_sym __p_class_syms[];
libresolv_hidden_proto (__p_class_syms)
const struct res_sym __p_class_syms[] = {
  {C_IN,    (char *) "IN"},
  {C_CHAOS, (char *) "CHAOS"},
  {C_HS,    (char *) "HS"},
  {C_HS,    (char *) "HESIOD"},
  {C_ANY,   (char *) "ANY"},
  {C_NONE,  (char *) "NONE"},
  {C_IN, NULL, NULL}
};
libresolv_hidden_data_def (__p_class_syms)

/*
 * Names of message sections.
 */
const struct res_sym __p_default_section_syms[] attribute_hidden = {
  {ns_s_qd, (char *) "QUERY"},
  {ns_s_an, (char *) "ANSWER"},
  {ns_s_ns, (char *) "AUTHORITY"},
  {ns_s_ar, (char *) "ADDITIONAL"},
  {0, NULL, NULL}
};

const struct res_sym __p_update_section_syms[] attribute_hidden = {
  {S_ZONE,   (char *) "ZONE"},
  {S_PREREQ, (char *) "PREREQUISITE"},
  {S_UPDATE, (char *) "UPDATE"},
  {S_ADDT,   (char *) "ADDITIONAL"},
  {0, NULL, NULL}
};

/*
 * Names of RR types and qtypes.  The list is incomplete because its
 * size is part of the ABI.
 */
extern const struct res_sym __p_type_syms[];
libresolv_hidden_proto (__p_type_syms)
const struct res_sym __p_type_syms[] = {
  {ns_t_a,      (char *) "A",     (char *) "address"},
  {ns_t_ns,     (char *) "NS",    (char *) "name server"},
  {ns_t_md,     (char *) "MD",    (char *) "mail destination (deprecated)"},
  {ns_t_mf,     (char *) "MF",    (char *) "mail forwarder (deprecated)"},
  {ns_t_cname,  (char *) "CNAME", (char *) "canonical name"},
  {ns_t_soa,    (char *) "SOA",   (char *) "start of authority"},
  {ns_t_mb,     (char *) "MB",    (char *) "mailbox"},
  {ns_t_mg,     (char *) "MG",    (char *) "mail group member"},
  {ns_t_mr,     (char *) "MR",    (char *) "mail rename"},
  {ns_t_null,   (char *) "NULL",  (char *) "null"},
  {ns_t_wks,    (char *) "WKS",   (char *) "well-known service (deprecated)"},
  {ns_t_ptr,    (char *) "PTR",   (char *) "domain name pointer"},
  {ns_t_hinfo,  (char *) "HINFO", (char *) "host information"},
  {ns_t_minfo,  (char *) "MINFO", (char *) "mailbox information"},
  {ns_t_mx,     (char *) "MX",    (char *) "mail exchanger"},
  {ns_t_txt,    (char *) "TXT",   (char *) "text"},
  {ns_t_rp,     (char *) "RP",    (char *) "responsible person"},
  {ns_t_afsdb,  (char *) "AFSDB", (char *) "DCE or AFS server"},
  {ns_t_x25,    (char *) "X25",   (char *) "X25 address"},
  {ns_t_isdn,   (char *) "ISDN",  (char *) "ISDN address"},
  {ns_t_rt,     (char *) "RT",    (char *) "router"},
  {ns_t_nsap,   (char *) "NSAP",  (char *) "nsap address"},
  {ns_t_nsap_ptr, (char *) "NSAP_PTR", (char *) "domain name pointer"},
  {ns_t_sig,    (char *) "SIG",   (char *) "signature"},
  {ns_t_key,    (char *) "KEY",   (char *) "key"},
  {ns_t_px,     (char *) "PX",    (char *) "mapping information"},
  {ns_t_gpos,   (char *) "GPOS",
   (char *) "geographical position (withdrawn)"},
  {ns_t_aaaa,   (char *) "AAAA",  (char *) "IPv6 address"},
  {ns_t_loc,    (char *) "LOC",   (char *) "location"},
  {ns_t_nxt,    (char *) "NXT",   (char *) "next valid name (unimplemented)"},
  {ns_t_eid,    (char *) "EID",   (char *) "endpoint identifier (unimplemented)"},
  {ns_t_nimloc, (char *) "NIMLOC", (char *) "NIMROD locator (unimplemented)"},
  {ns_t_srv,    (char *) "SRV",   (char *) "server selection"},
  {ns_t_atma,   (char *) "ATMA",  (char *) "ATM address (unimplemented)"},
  {ns_t_dname,  (char *) "DNAME", (char *) "Non-terminal DNAME (for IPv6)"},
  {ns_t_tsig,   (char *) "TSIG",  (char *) "transaction signature"},
  {ns_t_ixfr,   (char *) "IXFR",  (char *) "incremental zone transfer"},
  {ns_t_axfr,   (char *) "AXFR",  (char *) "zone transfer"},
  {ns_t_mailb,  (char *) "MAILB", (char *) "mailbox-related data (deprecated)"},
  {ns_t_maila,  (char *) "MAILA", (char *) "mail agent (deprecated)"},
  {ns_t_naptr,  (char *) "NAPTR", (char *) "URN Naming Authority"},
  {ns_t_kx,     (char *) "KX",    (char *) "Key Exchange"},
  {ns_t_cert,   (char *) "CERT",  (char *) "Certificate"},
  {ns_t_any,    (char *) "ANY",   (char *) "\"any\""},
  {0, NULL, NULL},		/* Padding to preserve ABI.  */
  {0, NULL, NULL}
};
libresolv_hidden_data_def (__p_type_syms)

/*
 * Names of DNS rcodes.
 */
const struct res_sym __p_rcode_syms[] attribute_hidden = {
  {ns_r_noerror,  (char *) "NOERROR",  (char *) "no error"},
  {ns_r_formerr,  (char *) "FORMERR",  (char *) "format error"},
  {ns_r_servfail, (char *) "SERVFAIL", (char *) "server failed"},
  {ns_r_nxdomain, (char *) "NXDOMAIN", (char *) "no such domain name"},
  {ns_r_notimpl,  (char *) "NOTIMP",   (char *) "not implemented"},
  {ns_r_refused,  (char *) "REFUSED",  (char *) "refused"},
  {ns_r_yxdomain, (char *) "YXDOMAIN", (char *) "domain name exists"},
  {ns_r_yxrrset,  (char *) "YXRRSET",  (char *) "rrset exists"},
  {ns_r_nxrrset,  (char *) "NXRRSET",  (char *) "rrset doesn't exist"},
  {ns_r_notauth,  (char *) "NOTAUTH",  (char *) "not authoritative"},
  {ns_r_notzone,  (char *) "NOTZONE",  (char *) "Not in zone"},
  {ns_r_max,      (char *) "",         (char *) ""},
  {ns_r_badsig,   (char *) "BADSIG",   (char *) "bad signature"},
  {ns_r_badkey,   (char *) "BADKEY",   (char *) "bad key"},
  {ns_r_badtime,  (char *) "BADTIME",  (char *) "bad time"},
  {0, NULL, NULL}
};

int
sym_ston(const struct res_sym *syms, const char *name, int *success) {
	for ((void)NULL; syms->name != 0; syms++) {
		if (strcasecmp (name, syms->name) == 0) {
			if (success)
				*success = 1;
			return (syms->number);
		}
	}
	if (success)
		*success = 0;
	return (syms->number);		/* The default value. */
}

const char *
sym_ntos(const struct res_sym *syms, int number, int *success) {
	static char unname[20];

	for ((void)NULL; syms->name != 0; syms++) {
		if (number == syms->number) {
			if (success)
				*success = 1;
			return (syms->name);
		}
	}

	sprintf(unname, "%d", number);		/* XXX nonreentrant */
	if (success)
		*success = 0;
	return (unname);
}
libresolv_hidden_def (sym_ntos)

const char *
sym_ntop(const struct res_sym *syms, int number, int *success) {
	static char unname[20];

	for ((void)NULL; syms->name != 0; syms++) {
		if (number == syms->number) {
			if (success)
				*success = 1;
			return (syms->humanname);
		}
	}
	sprintf(unname, "%d", number);		/* XXX nonreentrant */
	if (success)
		*success = 0;
	return (unname);
}

/*
 * Return a string for the type.
 */
const char *
p_type(int type) {
	return (sym_ntos(__p_type_syms, type, (int *)0));
}
libresolv_hidden_def (p_type)

/*
 * Return a string for the type.
 */
static const char *
p_section(int section, int opcode) {
	const struct res_sym *symbols;

	switch (opcode) {
	case ns_o_update:
		symbols = __p_update_section_syms;
		break;
	default:
		symbols = __p_default_section_syms;
		break;
	}
	return (sym_ntos(symbols, section, (int *)0));
}

/*
 * Return a mnemonic for class.
 */
const char *
p_class(int class) {
	return (sym_ntos(__p_class_syms, class, (int *)0));
}
libresolv_hidden_def (p_class)

/*
 * Return a mnemonic for an option
 */
const char *
p_option(u_long option) {
	static char nbuf[40];

	switch (option) {
	case RES_INIT:		return "init";
	case RES_DEBUG:		return "debug";
	case RES_USEVC:		return "use-vc";
	case RES_IGNTC:		return "igntc";
	case RES_RECURSE:	return "recurs";
	case RES_DEFNAMES:	return "defnam";
	case RES_STAYOPEN:	return "styopn";
	case RES_DNSRCH:	return "dnsrch";
	case RES_NOALIASES:	return "noaliases";
	case RES_ROTATE:	return "rotate";
	case RES_USE_EDNS0:	return "edns0";
	case RES_SNGLKUP:	return "single-request";
	case RES_SNGLKUPREOP:	return "single-request-reopen";
	case RES_USE_DNSSEC:	return "dnssec";
	case RES_NOTLDQUERY:	return "no-tld-query";
	case RES_NORELOAD:	return "no-reload";
	case RES_TRUSTAD:	return "trust-ad";
				/* XXX nonreentrant */
	default:		sprintf(nbuf, "?0x%lx?", (u_long)option);
				return (nbuf);
	}
}
libresolv_hidden_def (p_option)

/*
 * Return a mnemonic for a time to live.
 */
const char *
p_time(uint32_t value) {
	static char nbuf[40];		/* XXX nonreentrant */

	if (ns_format_ttl(value, nbuf, sizeof nbuf) < 0)
		sprintf(nbuf, "%u", value);
	return (nbuf);
}

/*
 * Return a string for the rcode.
 */
const char *
p_rcode(int rcode) {
	return (sym_ntos(__p_rcode_syms, rcode, (int *)0));
}
libresolv_hidden_def (p_rcode)

/*
 * routines to convert between on-the-wire RR format and zone file format.
 * Does not contain conversion to/from decimal degrees; divide or multiply
 * by 60*60*1000 for that.
 */

static const unsigned int poweroften[10]=
  { 1, 10, 100, 1000, 10000, 100000,
    1000000,10000000,100000000,1000000000};

/* takes an XeY precision/size value, returns a string representation. */
static const char *
precsize_ntoa (uint8_t prec)
{
	static char retbuf[sizeof "90000000.00"];	/* XXX nonreentrant */
	unsigned long val;
	int mantissa, exponent;

	mantissa = (int)((prec >> 4) & 0x0f) % 10;
	exponent = (int)((prec >> 0) & 0x0f) % 10;

	val = mantissa * poweroften[exponent];

	(void) sprintf(retbuf, "%ld.%.2ld", val/100, val%100);
	return (retbuf);
}

/* converts ascii size/precision X * 10**Y(cm) to 0xXY.  moves pointer. */
static uint8_t
precsize_aton (const char **strptr)
{
	unsigned int mval = 0, cmval = 0;
	uint8_t retval = 0;
	const char *cp;
	int exponent;
	int mantissa;

	cp = *strptr;

	while (isdigit(*cp))
		mval = mval * 10 + (*cp++ - '0');

	if (*cp == '.') {		/* centimeters */
		cp++;
		if (isdigit(*cp)) {
			cmval = (*cp++ - '0') * 10;
			if (isdigit(*cp)) {
				cmval += (*cp++ - '0');
			}
		}
	}
	cmval = (mval * 100) + cmval;

	for (exponent = 0; exponent < 9; exponent++)
		if (cmval < poweroften[exponent+1])
			break;

	mantissa = cmval / poweroften[exponent];
	if (mantissa > 9)
		mantissa = 9;

	retval = (mantissa << 4) | exponent;

	*strptr = cp;

	return (retval);
}

/* converts ascii lat/lon to unsigned encoded 32-bit number.  moves pointer. */
static uint32_t
latlon2ul (const char **latlonstrptr, int *which)
{
	const char *cp;
	uint32_t retval;
	int deg = 0, min = 0, secs = 0, secsfrac = 0;

	cp = *latlonstrptr;

	while (isdigit(*cp))
		deg = deg * 10 + (*cp++ - '0');

	while (isspace(*cp))
		cp++;

	if (!(isdigit(*cp)))
		goto fndhemi;

	while (isdigit(*cp))
		min = min * 10 + (*cp++ - '0');

	while (isspace(*cp))
		cp++;

	if (!(isdigit(*cp)))
		goto fndhemi;

	while (isdigit(*cp))
		secs = secs * 10 + (*cp++ - '0');

	if (*cp == '.') {		/* decimal seconds */
		cp++;
		if (isdigit(*cp)) {
			secsfrac = (*cp++ - '0') * 100;
			if (isdigit(*cp)) {
				secsfrac += (*cp++ - '0') * 10;
				if (isdigit(*cp)) {
					secsfrac += (*cp++ - '0');
				}
			}
		}
	}

	while (!isspace(*cp))	/* if any trailing garbage */
		cp++;

	while (isspace(*cp))
		cp++;

 fndhemi:
	switch (*cp) {
	case 'N': case 'n':
	case 'E': case 'e':
		retval = ((unsigned)1<<31)
			+ (((((deg * 60) + min) * 60) + secs) * 1000)
			+ secsfrac;
		break;
	case 'S': case 's':
	case 'W': case 'w':
		retval = ((unsigned)1<<31)
			- (((((deg * 60) + min) * 60) + secs) * 1000)
			- secsfrac;
		break;
	default:
		retval = 0;	/* invalid value -- indicates error */
		break;
	}

	switch (*cp) {
	case 'N': case 'n':
	case 'S': case 's':
		*which = 1;	/* latitude */
		break;
	case 'E': case 'e':
	case 'W': case 'w':
		*which = 2;	/* longitude */
		break;
	default:
		*which = 0;	/* error */
		break;
	}

	cp++;			/* skip the hemisphere */

	while (!isspace(*cp))	/* if any trailing garbage */
		cp++;

	while (isspace(*cp))	/* move to next field */
		cp++;

	*latlonstrptr = cp;

	return (retval);
}

/* converts a zone file representation in a string to an RDATA on-the-wire
 * representation. */
int
loc_aton (const char *ascii, u_char *binary)
{
	const char *cp, *maxcp;
	u_char *bcp;

	uint32_t latit = 0, longit = 0, alt = 0;
	uint32_t lltemp1 = 0, lltemp2 = 0;
	int altmeters = 0, altfrac = 0, altsign = 1;
	uint8_t hp = 0x16;	/* default = 1e6 cm = 10000.00m = 10km */
	uint8_t vp = 0x13;	/* default = 1e3 cm = 10.00m */
	uint8_t siz = 0x12;	/* default = 1e2 cm = 1.00m */
	int which1 = 0, which2 = 0;

	cp = ascii;
	maxcp = cp + strlen(ascii);

	lltemp1 = latlon2ul(&cp, &which1);

	lltemp2 = latlon2ul(&cp, &which2);

	switch (which1 + which2) {
	case 3:			/* 1 + 2, the only valid combination */
		if ((which1 == 1) && (which2 == 2)) { /* normal case */
			latit = lltemp1;
			longit = lltemp2;
		} else if ((which1 == 2) && (which2 == 1)) { /* reversed */
			longit = lltemp1;
			latit = lltemp2;
		} else {	/* some kind of brokenness */
			return (0);
		}
		break;
	default:		/* we didn't get one of each */
		return (0);
	}

	/* altitude */
	if (*cp == '-') {
		altsign = -1;
		cp++;
	}

	if (*cp == '+')
		cp++;

	while (isdigit(*cp))
		altmeters = altmeters * 10 + (*cp++ - '0');

	if (*cp == '.') {		/* decimal meters */
		cp++;
		if (isdigit(*cp)) {
			altfrac = (*cp++ - '0') * 10;
			if (isdigit(*cp)) {
				altfrac += (*cp++ - '0');
			}
		}
	}

	alt = (10000000 + (altsign * (altmeters * 100 + altfrac)));

	while (!isspace(*cp) && (cp < maxcp)) /* if trailing garbage or m */
		cp++;

	while (isspace(*cp) && (cp < maxcp))
		cp++;

	if (cp >= maxcp)
		goto defaults;

	siz = precsize_aton(&cp);

	while (!isspace(*cp) && (cp < maxcp))	/* if trailing garbage or m */
		cp++;

	while (isspace(*cp) && (cp < maxcp))
		cp++;

	if (cp >= maxcp)
		goto defaults;

	hp = precsize_aton(&cp);

	while (!isspace(*cp) && (cp < maxcp))	/* if trailing garbage or m */
		cp++;

	while (isspace(*cp) && (cp < maxcp))
		cp++;

	if (cp >= maxcp)
		goto defaults;

	vp = precsize_aton(&cp);

 defaults:

	bcp = binary;
	*bcp++ = (uint8_t) 0;	/* version byte */
	*bcp++ = siz;
	*bcp++ = hp;
	*bcp++ = vp;
	PUTLONG(latit,bcp);
	PUTLONG(longit,bcp);
	PUTLONG(alt,bcp);

	return (16);		/* size of RR in octets */
}

/* takes an on-the-wire LOC RR and formats it in a human readable format. */
const char *
loc_ntoa (const u_char *binary, char *ascii)
{
	static const char error[] = "?";
	static char tmpbuf[sizeof
"1000 60 60.000 N 1000 60 60.000 W -12345678.00m 90000000.00m 90000000.00m 90000000.00m"];
	const u_char *cp = binary;

	int latdeg, latmin, latsec, latsecfrac;
	int longdeg, longmin, longsec, longsecfrac;
	char northsouth, eastwest;
	int altmeters, altfrac, altsign;

	const uint32_t referencealt = 100000 * 100;

	int32_t latval, longval, altval;
	uint32_t templ;
	uint8_t sizeval, hpval, vpval, versionval;

	char *sizestr, *hpstr, *vpstr;

	versionval = *cp++;

	if (ascii == NULL)
		ascii = tmpbuf;

	if (versionval) {
		(void) sprintf(ascii, "; error: unknown LOC RR version");
		return (ascii);
	}

	sizeval = *cp++;

	hpval = *cp++;
	vpval = *cp++;

	GETLONG(templ, cp);
	latval = (templ - ((unsigned)1<<31));

	GETLONG(templ, cp);
	longval = (templ - ((unsigned)1<<31));

	GETLONG(templ, cp);
	if (templ < referencealt) { /* below WGS 84 spheroid */
		altval = referencealt - templ;
		altsign = -1;
	} else {
		altval = templ - referencealt;
		altsign = 1;
	}

	if (latval < 0) {
		northsouth = 'S';
		latval = -latval;
	} else
		northsouth = 'N';

	latsecfrac = latval % 1000;
	latval = latval / 1000;
	latsec = latval % 60;
	latval = latval / 60;
	latmin = latval % 60;
	latval = latval / 60;
	latdeg = latval;

	if (longval < 0) {
		eastwest = 'W';
		longval = -longval;
	} else
		eastwest = 'E';

	longsecfrac = longval % 1000;
	longval = longval / 1000;
	longsec = longval % 60;
	longval = longval / 60;
	longmin = longval % 60;
	longval = longval / 60;
	longdeg = longval;

	altfrac = altval % 100;
	altmeters = (altval / 100) * altsign;

	if ((sizestr = strdup(precsize_ntoa(sizeval))) == NULL)
		sizestr = (char *) error;
	if ((hpstr = strdup(precsize_ntoa(hpval))) == NULL)
		hpstr = (char *) error;
	if ((vpstr = strdup(precsize_ntoa(vpval))) == NULL)
		vpstr = (char *) error;

	sprintf(ascii,
	      "%d %.2d %.2d.%.3d %c %d %.2d %.2d.%.3d %c %d.%.2dm %sm %sm %sm",
		latdeg, latmin, latsec, latsecfrac, northsouth,
		longdeg, longmin, longsec, longsecfrac, eastwest,
		altmeters, altfrac, sizestr, hpstr, vpstr);

	if (sizestr != (char *) error)
		free(sizestr);
	if (hpstr != (char *) error)
		free(hpstr);
	if (vpstr != (char *) error)
		free(vpstr);

	return (ascii);
}
libresolv_hidden_def (loc_ntoa)


/* Return the number of DNS hierarchy levels in the name. */
int
dn_count_labels(const char *name) {
	int i, len, count;

	len = strlen(name);
	for (i = 0, count = 0; i < len; i++) {
		/* XXX need to check for \. or use named's nlabels(). */
		if (name[i] == '.')
			count++;
	}

	/* don't count initial wildcard */
	if (name[0] == '*')
		if (count)
			count--;

	/* don't count the null label for root. */
	/* if terminating '.' not found, must adjust */
	/* count to include last label */
	if (len > 0 && name[len-1] != '.')
		count++;
	return (count);
}
libresolv_hidden_def (__dn_count_labels)


#if SHLIB_COMPAT (libresolv, GLIBC_2_0, GLIBC_2_27)
/*
 * Make dates expressed in seconds-since-Jan-1-1970 easy to read.
 * SIG records are required to be printed like this, by the Secure DNS RFC.
 * This is an obsolescent function and does not handle dates outside the
 * signed 32-bit range.
 */
char *
__p_secstodate (u_long secs) {
	/* XXX nonreentrant */
	static char output[15];		/* YYYYMMDDHHMMSS and null */
	time_t clock = secs;
	struct tm *time;

	struct tm timebuf;
	/* The call to __gmtime_r can never produce a year overflowing
	   the range of int, given the check on SECS, but check for a
	   NULL return anyway to avoid a null pointer dereference in
	   case there are any other unspecified errors.  */
	if (secs > 0x7fffffff
	    || (time = __gmtime_r (&clock, &timebuf)) == NULL) {
		strcpy (output, "<overflow>");
		__set_errno (EOVERFLOW);
		return output;
	}
	time->tm_year += 1900;
	time->tm_mon += 1;
	/* The struct tm fields, given the above range check,
	   must have values that mean this sprintf exactly fills the
	   buffer.  But as of GCC 8 of 2017-11-21, GCC cannot tell
	   that, even given range checks on all fields with
	   __builtin_unreachable called for out-of-range values.  */
	DIAG_PUSH_NEEDS_COMMENT;
# if __GNUC_PREREQ (7, 0)
	DIAG_IGNORE_NEEDS_COMMENT (8, "-Wformat-overflow=");
# endif
	sprintf(output, "%04d%02d%02d%02d%02d%02d",
		time->tm_year, time->tm_mon, time->tm_mday,
		time->tm_hour, time->tm_min, time->tm_sec);
	DIAG_POP_NEEDS_COMMENT;
	return (output);
}
compat_symbol (libresolv, __p_secstodate, __p_secstodate, GLIBC_2_0);
#endif
