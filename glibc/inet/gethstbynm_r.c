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

#include <ctype.h>
#include <errno.h>
#include <netdb.h>
#include <string.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <resolv/res_hconf.h>

#define LOOKUP_TYPE	struct hostent
#define FUNCTION_NAME	gethostbyname
#define DATABASE_NAME	hosts
#define ADD_PARAMS	const char *name
#define ADD_VARIABLES	name
#define NEED_H_ERRNO	1
#define NEED__RES	1
#define POSTPROCESS \
  if (status == NSS_STATUS_SUCCESS)					      \
    _res_hconf_reorder_addrs (resbuf);

#define HANDLE_DIGITS_DOTS	1
#define HAVE_LOOKUP_BUFFER	1

/* Special name for the lookup function.  */
#define DB_LOOKUP_FCT __nss_hosts_lookup2

#include "../nss/getXXbyYY_r.c"
