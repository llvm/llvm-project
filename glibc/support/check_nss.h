/* Test verification functions for NSS- and DNS-related data.
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

#ifndef SUPPORT_CHECK_NSS_H
#define SUPPORT_CHECK_NSS_H

#include <netdb.h>
#include <sys/cdefs.h>

__BEGIN_DECLS

/* Compare the data structures against the expected values (which have
   to be formatted according to the support_format_* functions in
   <support/format_nss.h>).  If there is a difference, a delayed test
   failure is recorded, and a diff is written to standard output.  */
void check_addrinfo (const char *query_description,
                     struct addrinfo *, int ret, const char *expected);
void check_dns_packet (const char *query_description,
                       const unsigned char *, size_t, const char *expected);
void check_hostent (const char *query_description,
                    struct hostent *, const char *expected);
void check_netent (const char *query_description,
                   struct netent *, const char *expected);

__END_DECLS

#endif  /* SUPPORT_CHECK_NSS_H */
