/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper, <drepper@gnu.org>.

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
#include <locale.h>
#include <nl_types.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>

#include "catgetsinfo.h"


/* Open the catalog and return a descriptor for the catalog.  */
nl_catd
catopen (const char *cat_name, int flag)
{
  __nl_catd result;
  const char *env_var = NULL;
  const char *nlspath = NULL;
  char *tmp = NULL;

  if (strchr (cat_name, '/') == NULL)
    {
      if (flag == NL_CAT_LOCALE)
	/* Use the current locale setting for LC_MESSAGES.  */
	env_var = setlocale (LC_MESSAGES, NULL);
      else
	/* Use the LANG environment variable.  */
	env_var = getenv ("LANG");

      if (env_var == NULL || *env_var == '\0'
	  || (__libc_enable_secure && strchr (env_var, '/') != NULL))
	env_var = "C";

      nlspath = getenv ("NLSPATH");
      if (nlspath != NULL && *nlspath != '\0')
	{
	  /* Append the system dependent directory.  */
	  size_t len = strlen (nlspath) + 1 + sizeof NLSPATH;
	  tmp = malloc (len);

	  if (__glibc_unlikely (tmp == NULL))
	    return (nl_catd) -1;

	  __stpcpy (__stpcpy (__stpcpy (tmp, nlspath), ":"), NLSPATH);
	  nlspath = tmp;
	}
      else
	nlspath = NLSPATH;
    }

  result = (__nl_catd) malloc (sizeof (*result));
  if (result == NULL)
    {
      /* We cannot get enough memory.  */
      result = (nl_catd) -1;
    }
  else if (__open_catalog (cat_name, nlspath, env_var, result) != 0)
    {
      /* Couldn't open the file.  */
      free ((void *) result);
      result = (nl_catd) -1;
    }

  free (tmp);
  return (nl_catd) result;
}


/* Return message from message catalog.  */
char *
catgets (nl_catd catalog_desc, int set, int message, const char *string)
{
  __nl_catd catalog;
  size_t idx;
  size_t cnt;

  /* Be generous if catalog which failed to be open is used.  */
  if (catalog_desc == (nl_catd) -1 || ++set <= 0 || message < 0)
    return (char *) string;

  catalog = (__nl_catd) catalog_desc;

  idx = ((set * message) % catalog->plane_size) * 3;
  cnt = 0;
  do
    {
      if (catalog->name_ptr[idx + 0] == (uint32_t) set
	  && catalog->name_ptr[idx + 1] == (uint32_t) message)
	return (char *) &catalog->strings[catalog->name_ptr[idx + 2]];

      idx += catalog->plane_size * 3;
    }
  while (++cnt < catalog->plane_depth);

  __set_errno (ENOMSG);
  return (char *) string;
}


/* Return resources used for loaded message catalog.  */
int
catclose (nl_catd catalog_desc)
{
  __nl_catd catalog;

  /* Be generous if catalog which failed to be open is used.  */
  if (catalog_desc == (nl_catd) -1)
    {
      __set_errno (EBADF);
      return -1;
    }

  catalog = (__nl_catd) catalog_desc;

#ifdef _POSIX_MAPPED_FILES
  if (catalog->status == mmapped)
    __munmap ((void *) catalog->file_ptr, catalog->file_size);
  else
#endif	/* _POSIX_MAPPED_FILES */
    if (catalog->status == malloced)
      free ((void *) catalog->file_ptr);
    else
      {
	__set_errno (EBADF);
	return -1;
      }

  free ((void *) catalog);

  return 0;
}
