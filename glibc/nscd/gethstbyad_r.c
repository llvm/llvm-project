/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1996.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published
   by the Free Software Foundation; version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, see <https://www.gnu.org/licenses/>.  */

#include <netdb.h>


#define LOOKUP_TYPE	struct hostent
#define FUNCTION_NAME	gethostbyaddr2
#define FUNCTION2_NAME	gethostbyaddr
#define DATABASE_NAME	hosts
#define ADD_PARAMS	const void *addr, socklen_t len, int type
#define EXTRA_PARAMS	, int32_t *ttlp
#define ADD_VARIABLES	addr, len, type
#define EXTRA_VARIABLES , ttlp
#define NEED_H_ERRNO	1
#define NEED__RES	1

/* We are nscd, so we don't want to be talking to ourselves.  */
#undef	USE_NSCD

#include "../nss/getXXbyYY_r.c"


int
__gethostbyaddr_r (const void *addr, socklen_t len, int type,
		   struct hostent *result_buf, char *buf, size_t buflen,
		   struct hostent **result, int *h_errnop)
{
  return __gethostbyaddr2_r (addr, len, type, result_buf, buf, buflen,
			     result, h_errnop, NULL);
}
