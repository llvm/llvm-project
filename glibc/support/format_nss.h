/* String formatting functions for NSS- and DNS-related data.
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

#ifndef SUPPORT_FORMAT_NSS_H
#define SUPPORT_FORMAT_NSS_H

#include <netdb.h>
#include <sys/cdefs.h>

__BEGIN_DECLS

/* The following functions format their arguments as human-readable
   strings (which can span multiple lines).  The caller must free the
   returned buffer.  For NULL pointers or failure status arguments,
   error variables such as h_errno and errno are included in the
   result.  */
char *support_format_address_family (int);
char *support_format_addrinfo (struct addrinfo *, int ret);
char *support_format_dns_packet (const unsigned char *buffer, size_t length);
char *support_format_herrno (int);
char *support_format_hostent (struct hostent *);
char *support_format_netent (struct netent *);

__END_DECLS

#endif  /* SUPPORT_FORMAT_NSS_H */
