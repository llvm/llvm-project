/* Internal routines for nss_dns.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#ifndef _NSS_DNS_H
#define _NSS_DNS_H

#include <nss.h>

NSS_DECLARE_MODULE_FUNCTIONS (dns)

libc_hidden_proto (_nss_dns_getcanonname_r)
libc_hidden_proto (_nss_dns_gethostbyaddr2_r)
libc_hidden_proto (_nss_dns_gethostbyaddr_r)
libc_hidden_proto (_nss_dns_gethostbyname2_r)
libc_hidden_proto (_nss_dns_gethostbyname3_r)
libc_hidden_proto (_nss_dns_gethostbyname4_r)
libc_hidden_proto (_nss_dns_gethostbyname_r)
libc_hidden_proto (_nss_dns_getnetbyaddr_r)
libc_hidden_proto (_nss_dns_getnetbyname_r)

void __nss_dns_functions (nss_module_functions_untyped pointers)
  attribute_hidden;

#endif
