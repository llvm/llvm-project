/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
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

#include <netdb.h>


#define LOOKUP_TYPE	struct hostent
#define SETFUNC_NAME	sethostent
#define	GETFUNC_NAME	gethostent
#define	ENDFUNC_NAME	endhostent
#define DATABASE_NAME	hosts
#define STAYOPEN	int stayopen
#define STAYOPEN_VAR	stayopen
#define NEED_H_ERRNO	1
#define NEED__RES	1

/* Special name for the lookup function.  */
#define DB_LOOKUP_FCT __nss_hosts_lookup2

#include "../nss/getXXent_r.c"
