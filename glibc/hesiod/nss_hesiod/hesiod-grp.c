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

#include <ctype.h>
#include <errno.h>
#include <grp.h>
#include <hesiod.h>
#include <nss.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>

NSS_DECLARE_MODULE_FUNCTIONS (hesiod)

/* Get the declaration of the parser function.  */
#define ENTNAME grent
#define STRUCTURE group
#define EXTERN_PARSER
#include <nss/nss_files/files-parse.c>

enum nss_status
_nss_hesiod_setgrent (int stayopen)
{
  return NSS_STATUS_SUCCESS;
}

enum nss_status
_nss_hesiod_endgrent (void)
{
  return NSS_STATUS_SUCCESS;
}

static enum nss_status
lookup (const char *name, const char *type, struct group *grp,
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

  parse_res = _nss_files_parse_grent (buffer, grp, data, buflen, errnop);
  if (parse_res < 1)
    {
      __set_errno (olderr);
      return parse_res == -1 ? NSS_STATUS_TRYAGAIN : NSS_STATUS_NOTFOUND;
    }

  return NSS_STATUS_SUCCESS;
}

enum nss_status
_nss_hesiod_getgrnam_r (const char *name, struct group *grp,
			char *buffer, size_t buflen, int *errnop)
{
  return lookup (name, "group", grp, buffer, buflen, errnop);
}

enum nss_status
_nss_hesiod_getgrgid_r (gid_t gid, struct group *grp,
			char *buffer, size_t buflen, int *errnop)
{
  char gidstr[21];	/* We will probably never have a gid_t with more
			   than 64 bits.  */

  snprintf (gidstr, sizeof gidstr, "%d", gid);

  return lookup (gidstr, "gid", grp, buffer, buflen, errnop);
}

static int
internal_gid_in_list (const gid_t *list, const gid_t g, long int len)
{
  while (len > 0)
    {
      if (*list == g)
	return 1;
      --len;
      ++list;
    }
  return 0;
}

static enum nss_status
internal_gid_from_group (void *context, const char *groupname, gid_t *group)
{
  char **grp_res;
  enum nss_status status = NSS_STATUS_NOTFOUND;

  grp_res = hesiod_resolve (context, groupname, "group");
  if (grp_res != NULL && *grp_res != NULL)
    {
      char *p = *grp_res;

      /* Skip to third field.  */
      while (*p != '\0' && *p != ':')
	++p;
      if (*p != '\0')
	++p;
      while (*p != '\0' && *p != ':')
	++p;
      if (*p != '\0')
	{
	  char *endp;
	  char *q = ++p;
	  long int val;

	  while (*q != '\0' && *q != ':')
	    ++q;

	  val = strtol (p, &endp, 10);
	  if (sizeof (gid_t) == sizeof (long int) || (gid_t) val == val)
	    {
	      *group = val;
	      if (endp == q && endp != p)
		status = NSS_STATUS_SUCCESS;
	    }
        }
      hesiod_free_list (context, grp_res);
    }
  return status;
}

enum nss_status
_nss_hesiod_initgroups_dyn (const char *user, gid_t group, long int *start,
			    long int *size, gid_t **groupsp, long int limit,
			    int *errnop)
{
  enum nss_status status = NSS_STATUS_SUCCESS;
  char **list = NULL;
  char *p;
  void *context;
  gid_t *groups = *groupsp;
  int save_errno;

  if (hesiod_init (&context) < 0)
    return NSS_STATUS_UNAVAIL;

  list = hesiod_resolve (context, user, "grplist");

  if (list == NULL)
    {
      hesiod_end (context);
      return errno == ENOENT ? NSS_STATUS_NOTFOUND : NSS_STATUS_UNAVAIL;
    }

  save_errno = errno;

  p = *list;
  while (*p != '\0')
    {
      char *endp;
      char *q;
      long int val;

      status = NSS_STATUS_NOTFOUND;

      q = p;
      while (*q != '\0' && *q != ':' && *q != ',')
	++q;

      if (*q != '\0')
	*q++ = '\0';

      __set_errno (0);
      val = strtol (p, &endp, 10);
      /* Test whether the number is representable in a variable of
         type `gid_t'.  If not ignore the number.  */
      if ((sizeof (gid_t) == sizeof (long int) || (gid_t) val == val)
	  && errno == 0)
	{
	  if (*endp == '\0' && endp != p)
	    {
	      group = val;
	      status = NSS_STATUS_SUCCESS;
	    }
	  else
	    status = internal_gid_from_group (context, p, &group);

	  if (status == NSS_STATUS_SUCCESS
	      && !internal_gid_in_list (groups, group, *start))
	    {
	      if (__glibc_unlikely (*start == *size))
		{
		  /* Need a bigger buffer.  */
		  gid_t *newgroups;
		  long int newsize;

		  if (limit > 0 && *size == limit)
		    /* We reached the maximum.  */
		    goto done;

		  if (limit <= 0)
		    newsize = 2 * *size;
		  else
		    newsize = MIN (limit, 2 * *size);

		  newgroups = realloc (groups, newsize * sizeof (*groups));
		  if (newgroups == NULL)
		    goto done;
		  *groupsp = groups = newgroups;
		  *size = newsize;
		}

	      groups[(*start)++] = group;
	    }
	}

      p = q;
    }

  __set_errno (save_errno);

 done:
  hesiod_free_list (context, list);
  hesiod_end (context);

  return NSS_STATUS_SUCCESS;
}
