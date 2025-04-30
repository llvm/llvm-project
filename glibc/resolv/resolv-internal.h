/* libresolv interfaces for internal use across glibc.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#ifndef _RESOLV_INTERNAL_H
#define _RESOLV_INTERNAL_H 1

#include <resolv.h>
#include <stdbool.h>

/* Resolver flags.  Used for _flags in struct __res_state.  */
#define RES_F_VC        0x00000001 /* Socket is TCP.  */
#define RES_F_CONN      0x00000002 /* Socket is connected.  */
#define RES_F_EDNS0ERR  0x00000004 /* EDNS0 caused errors.  */

/* Legacy function.  This needs to be removed once all NSS modules
   have been adjusted.  */
static inline bool
res_use_inet6 (void)
{
  return false;
}

enum
  {
    /* The advertized EDNS buffer size.  The value 1200 is derived
       from the IPv6 minimum MTU (1280 bytes) minus some arbitrary
       space for tunneling overhead.  If the DNS server does not react
       to ICMP Fragmentation Needed But DF Set messages, this should
       avoid all UDP fragments on current networks.  Avoiding UDP
       fragments is desirable because it prevents fragmentation-based
       spoofing attacks because the randomness in a DNS packet is
       concentrated in the first fragment (with the headers) and does
       not protect subsequent fragments.  */
    RESOLV_EDNS_BUFFER_SIZE = 1200,
  };

struct resolv_context;

/* Internal function for implementing res_nmkquery and res_mkquery.
   Also used by __res_context_query.  */
int __res_context_mkquery (struct resolv_context *, int op, const char *dname,
                           int class, int type, const unsigned char *data,
                           unsigned char *buf, int buflen);
libc_hidden_proto (__res_context_mkquery)

/* Main resolver query function for use within glibc.  */
int __res_context_search (struct resolv_context *, const char *, int, int,
                          unsigned char *, int, unsigned char **,
                          unsigned char **, int *, int *, int *);
libc_hidden_proto (__res_context_search)

/* Main resolver query function for use within glibc.  */
int __res_context_query (struct resolv_context *, const char *, int, int,
                         unsigned char *, int, unsigned char **,
                         unsigned char **, int *, int *, int *);
libc_hidden_proto (__res_context_query)

/* Internal function used to implement the query and search
   functions.  */
int __res_context_send (struct resolv_context *, const unsigned char *, int,
                        const unsigned char *, int, unsigned char *,
                        int, unsigned char **, unsigned char **,
                        int *, int *, int *);
libc_hidden_proto (__res_context_send)

/* Internal function similar to res_hostalias.  */
const char *__res_context_hostalias (struct resolv_context *,
                                     const char *, char *, size_t);
libc_hidden_proto (__res_context_hostalias);

/* Add an OPT record to a DNS query.  */
int __res_nopt (struct resolv_context *, int n0,
                unsigned char *buf, int buflen, int anslen);
libc_hidden_proto (__res_nopt)

/* Convert from presentation format (which usually means ASCII
   printable) to network format (which is usually some kind of binary
   format).  The input is in the range [SRC, SRC + SRCLEN).  The
   output is written to DST (which has to be 4 or 16 bytes long,
   depending on AF).  Return 0 for invalid input, 1 for success, -1
   for an invalid address family.  */
int __inet_pton_length (int af, const char *src, size_t srclen, void *);
libc_hidden_proto (__inet_pton_length)

/* Called as part of the thread shutdown sequence.  */
void __res_thread_freeres (void) attribute_hidden;

/* The Linux kernel does not enable all ICMP messages on a UDP socket
   by default.  A call this function enables full error reporting for
   the socket FD.  FAMILY must be AF_INET or AF_INET6.  Returns 0 on
   success, -1 on failure.  */
int __res_enable_icmp (int family, int fd) attribute_hidden;


/* Returns the name server address for the indicated index.  */
struct sockaddr *__res_get_nsaddr (res_state statp, unsigned int n);
libc_hidden_proto (__res_get_nsaddr)


#endif  /* _RESOLV_INTERNAL_H */
