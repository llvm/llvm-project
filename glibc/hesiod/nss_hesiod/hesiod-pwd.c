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
#include <pwd.h>
#include <nss.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

NSS_DECLARE_MODULE_FUNCTIONS (hesiod)

/* Get the declaration of the parser function.  */
#define ENTNAME pwent
#define STRUCTURE passwd
#define EXTERN_PARSER
#include <nss/nss_files/files-parse.c>

enum nss_status
_nss_hesiod_setpwent (int stayopen)
{
  return NSS_STATUS_SUCCESS;
}

enum nss_status
_nss_hesiod_endpwent (void)
{
  return NSS_STATUS_SUCCESS;
}

static enum nss_status
lookup (const char *name, const char *type, struct passwd *pwd,
	char *buffer, size_t buflen, int *errnop)
{
  struct parser_data *data = (void *) buffer;
  size_t linebuflen;
  void *context;
  char **list;
  int parse_res;
  size_t len;
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
  len = strlen (*list) + 1;
  if (linebuflen < len)
    {
      hesiod_free_list (context, list);
      hesiod_end (context);
      *errnop = ERANGE;
      return NSS_STATUS_TRYAGAIN;
    }

  memcpy (data->linebuffer, *list, len);
  hesiod_free_list (context, list);
  hesiod_end (context);

  parse_res = _nss_files_parse_pwent (buffer, pwd, data, buflen, errnop);
  if (parse_res < 1)
    {
      __set_errno (olderr);
      return parse_res == -1 ? NSS_STATUS_TRYAGAIN : NSS_STATUS_NOTFOUND;
    }

  return NSS_STATUS_SUCCESS;
}

enum nss_status
_nss_hesiod_getpwnam_r (const char *name, struct passwd *pwd,
			char *buffer, size_t buflen, int *errnop)
{
  return lookup (name, "passwd", pwd, buffer, buflen, errnop);
}

enum nss_status
_nss_hesiod_getpwuid_r (uid_t uid, struct passwd *pwd,
			char *buffer, size_t buflen, int *errnop)
{
  char uidstr[21];	/* We will probably never have a gid_t with more
			   than 64 bits.  */

  snprintf (uidstr, sizeof uidstr, "%d", uid);

  return lookup (uidstr, "uid", pwd, buffer, buflen, errnop);
}
