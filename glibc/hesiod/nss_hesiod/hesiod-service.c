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

/* Hesiod uses a format for service entries that differs from the
   traditional format.  We therefore declare our own parser.  */

#define ENTNAME servent

struct servent_data {};

#define TRAILING_LIST_MEMBER		s_aliases
#define TRAILING_LIST_SEPARATOR_P	isspace
#include <nss/nss_files/files-parse.c>
#define ISSC_OR_SPACE(c)	((c) ==  ';' || isspace (c))
LINE_PARSER
("#",
 STRING_FIELD (result->s_name, ISSC_OR_SPACE, 1);
 STRING_FIELD (result->s_proto, ISSC_OR_SPACE, 1);
 INT_FIELD (result->s_port, ISSC_OR_SPACE, 10, 0, htons);
 )

enum nss_status
_nss_hesiod_setservent (int stayopen)
{
  return NSS_STATUS_SUCCESS;
}

enum nss_status
_nss_hesiod_endservent (void)
{
  return NSS_STATUS_SUCCESS;
}

static enum nss_status
lookup (const char *name, const char *type, const char *protocol,
	struct servent *serv, char *buffer, size_t buflen, int *errnop)
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

      parse_res = parse_line (buffer, serv, data, buflen, errnop);
      if (parse_res == -1)
	{
	  hesiod_free_list (context, list);
	  hesiod_end (context);
	  return NSS_STATUS_TRYAGAIN;
	}

      if (parse_res > 0)
	found = protocol == NULL || strcasecmp (serv->s_proto, protocol) == 0;

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
_nss_hesiod_getservbyname_r (const char *name, const char *protocol,
			     struct servent *serv,
			     char *buffer, size_t buflen, int *errnop)
{
  return lookup (name, "service", protocol, serv, buffer, buflen, errnop);
}

enum nss_status
_nss_hesiod_getservbyport_r (const int port, const char *protocol,
			     struct servent *serv,
			     char *buffer, size_t buflen, int *errnop)
{
  char portstr[6];	    /* Port numbers are restricted to 16 bits. */

  snprintf (portstr, sizeof portstr, "%d", ntohs (port));

  return lookup (portstr, "port", protocol, serv, buffer, buflen, errnop);
}
