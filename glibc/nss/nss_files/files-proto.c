/* Protocols file parser in nss_files module.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
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
#include <nss.h>

#define ENTNAME		protoent
#define DATABASE	"protocols"

struct protoent_data {};

#define TRAILING_LIST_MEMBER		p_aliases
#define TRAILING_LIST_SEPARATOR_P	isspace
#include "files-parse.c"
LINE_PARSER
("#",
 STRING_FIELD (result->p_name, isspace, 1);
 INT_FIELD (result->p_proto, isspace, 1, 10,);
 )

#include GENERIC

DB_LOOKUP (protobyname, '.', 0, ("%s", name),
	   LOOKUP_NAME (p_name, p_aliases),
	   const char *name)

DB_LOOKUP (protobynumber, '=', 20, ("%zd", (ssize_t) proto),
	   {
	     if (result->p_proto == proto)
	       break;
	   }, int proto)
