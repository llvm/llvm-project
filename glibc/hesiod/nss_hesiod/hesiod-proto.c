/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Mark Kettenis <kettenis@phys.uva.nl>, 1997.

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

#include <errno.h>
#include <hesiod.h>
#include <netdb.h>
#include <netinet/in.h>
#include <nss.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

NSS_DECLARE_MODULE_FUNCTIONS (hesiod)

/* Declare a parser for Hesiod protocol entries.  Although the format
   of the entries is identical to those in /etc/protocols, here is no
   predefined parser for us to use.  */

#define ENTNAME protoent

struct protoent_data {};

#define TRAILING_LIST_MEMBER		p_aliases
#define TRAILING_LIST_SEPARATOR_P	isspace
#include <nss/nss_files/files-parse.c>
LINE_PARSER
("#",
 STRING_FIELD (result->p_name, isspace, 1);
 INT_FIELD (result->p_proto, isspace, 1, 10,);
 )

enum nss_status
_nss_hesiod_setprotoent (int stayopen)
{
  return NSS_STATUS_SUCCESS;
}

enum nss_status
_nss_hesiod_endprotoent (void)
{
  return NSS_STATUS_SUCCESS;
}

static enum nss_status
lookup (const char *name, const char *type, struct protoent *proto,
	char *buffer, size_t buflen, int *errnop)
{
  struct parser_data *data = (void *) buffer;
  size_t linebuflen;
  void *context;
  char **list, **item;
  int parse_res;
  int found;
  int olderr = errno;

  if (hesiod_init (&context) < 0)
    return NSS_STATUS_UNAVAIL;

  list = hesiod_resolve (context, name, type);
  if (list == NULL)
    {
      int err = errno;
      hesiod_end (context);
      __set_errno (olderr);
      return err == ENOENT ? NSS_STATUS_NOTFOUND : NSS_STATUS_UNAVAIL;
    }

  linebuflen = buffer + buflen - data->linebuffer;

  item = list;
  found = 0;
  do
    {
      size_t len = strlen (*item) + 1;

      if (linebuflen < len)
	{
	  hesiod_free_list (context, list);
	  hesiod_end (context);
	  *errnop = ERANGE;
	  return NSS_STATUS_TRYAGAIN;
	}

      memcpy (data->linebuffer, *item, len);

      parse_res = parse_line (buffer, proto, data, buflen, errnop);
      if (parse_res == -1)
	{
	  hesiod_free_list (context, list);
	  hesiod_end (context);
	  return NSS_STATUS_TRYAGAIN;
	}

      if (parse_res > 0)
	found = 1;

      ++item;
    }
  while (*item != NULL && !found);

  hesiod_free_list (context, list);
  hesiod_end (context);

  if (found == 0)
    {
      __set_errno (olderr);
      return NSS_STATUS_NOTFOUND;
    }

  return NSS_STATUS_SUCCESS;
}

enum nss_status
_nss_hesiod_getprotobyname_r (const char *name, struct protoent *proto,
			      char *buffer, size_t buflen, int *errnop)
{
  return lookup (name, "protocol", proto, buffer, buflen, errnop);
}

enum nss_status
_nss_hesiod_getprotobynumber_r (const int protocol, struct protoent *proto,
				char *buffer, size_t buflen, int *errnop)
{
  char protostr[21];

  snprintf (protostr, sizeof protostr, "%d", protocol);

  return lookup (protostr, "protonum", proto, buffer, buflen, errnop);
}
