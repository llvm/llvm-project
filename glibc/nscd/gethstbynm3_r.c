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

#include <ctype.h>
#include <errno.h>
#include <netdb.h>
#include <string.h>
#include <arpa/inet.h>
#include <netinet/in.h>


#define LOOKUP_TYPE	struct hostent
#define FUNCTION_NAME	gethostbyname3
#define FUNCTION2_NAME	gethostbyname2
#define DATABASE_NAME	hosts
#define ADD_PARAMS	const char *name, int af
#define EXTRA_PARAMS	, int32_t *ttlp, char **canonp
#define ADD_VARIABLES	name, af
#define EXTRA_VARIABLES	, ttlp, canonp
#define NEED_H_ERRNO	1
#define NEED__RES       1

#define HANDLE_DIGITS_DOTS	1
#define HAVE_LOOKUP_BUFFER	1
#define HAVE_AF			1

/* We are nscd, so we don't want to be talking to ourselves.  */
#undef	USE_NSCD

#include "../nss/getXXbyYY_r.c"


int
__gethostbyname2_r (const char *name, int af, struct hostent *ret, char *buf,
		    size_t buflen, struct hostent **result, int *h_errnop)
{
  return __gethostbyname3_r (name, af, ret, buf, buflen, result, h_errnop,
			     NULL, NULL);
}
