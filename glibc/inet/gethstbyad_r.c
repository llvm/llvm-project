/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1996.

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

#include <netdb.h>
#include <string.h>
#include <resolv/res_hconf.h>

#define LOOKUP_TYPE	struct hostent
#define FUNCTION_NAME	gethostbyaddr
#define DATABASE_NAME	hosts
#define ADD_PARAMS	const void *addr, socklen_t len, int type
#define ADD_VARIABLES	addr, len, type
#define NEED_H_ERRNO	1
#define NEED__RES	1
/* If the addr parameter is the IPv6 unspecified address no query must
   be performed.  */
#define PREPROCESS \
  if (len == sizeof (struct in6_addr)					      \
      && __builtin_expect (memcmp (&__in6addr_any, addr,		      \
				   sizeof (struct in6_addr)), 1) == 0)	      \
    {									      \
      *h_errnop = HOST_NOT_FOUND;					      \
      *result = NULL;							      \
      return ENOENT;							      \
    }
#define POSTPROCESS \
  if (status == NSS_STATUS_SUCCESS)					      \
    {									      \
      _res_hconf_reorder_addrs (resbuf);				      \
      _res_hconf_trim_domains (resbuf);					      \
    }

/* Special name for the lookup function.  */
#define DB_LOOKUP_FCT __nss_hosts_lookup2

#include "../nss/getXXbyYY_r.c"
