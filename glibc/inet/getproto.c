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


#define LOOKUP_TYPE	struct protoent
#define FUNCTION_NAME	getprotobynumber
#define DATABASE_NAME	protocols
#define ADD_PARAMS	int proto
#define ADD_VARIABLES	proto
#define BUFLEN		1024

/* There is no nscd support for the protocols file.  */
#undef	USE_NSCD

#include "../nss/getXXbyYY.c"
