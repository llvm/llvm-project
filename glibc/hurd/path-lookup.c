/* Filename lookup using a search path
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Written by Miles Bader <miles@gnu.ai.mit.edu>

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

#include <string.h>
#include <hurd.h>
#include <hurd/lookup.h>

/* If FILE_NAME contains a '/', or PATH is NULL, call FUN with FILE_NAME, and
   return the result (if PREFIXED_NAME is non-NULL, setting *PREFIXED_NAME to
   NULL).  Otherwise, call FUN repeatedly with FILE_NAME prefixed with each
   successive `:' separated element of PATH, returning whenever FUN returns
   0 (if PREFIXED_NAME is non-NULL, setting *PREFIXED_NAME to the resulting
   prefixed path).  If FUN never returns 0, return the first non-ENOENT
   return value, or ENOENT if there is none.  */
error_t
file_name_path_scan (const char *file_name, const char *path,
		     error_t (*fun)(const char *name),
		     char **prefixed_name)
{
  if (path == NULL || strchr (file_name, '/'))
    {
      if (prefixed_name)
	*prefixed_name = 0;
      return (*fun)(file_name);
    }
  else
    {
      error_t real_err = 0;
      size_t file_name_len = strlen (file_name);

      for (;;)
	{
	  error_t err;
	  const char *next = strchr (path, ':') ?: path + strlen (path);
	  size_t pfx_len = next - path;
	  char pfxed_name[pfx_len + 2 + file_name_len + 1];

	  if (pfx_len == 0)
	    pfxed_name[pfx_len++] = '.';
	  else
	    memcpy (pfxed_name, path, pfx_len);
	  if (pfxed_name[pfx_len - 1] != '/')
	    pfxed_name[pfx_len++] = '/';
	  memcpy (pfxed_name + pfx_len, file_name, file_name_len + 1);

	  err = (*fun)(pfxed_name);
	  if (err == 0)
	    {
	      if (prefixed_name)
		*prefixed_name = __strdup (pfxed_name);
	      return 0;
	    }
	  if (!real_err && err != ENOENT)
	    real_err = err;

	  if (*next == '\0')
	    return real_err ?: ENOENT;
	  else
	    path = next + 1;
	}
    }
}

/* Lookup FILE_NAME and return the node opened with FLAGS & MODE in result
   (see hurd_file_name_lookup for details), but a simple filename (without
   any directory prefixes) will be consecutively prefixed with the pathnames
   in the `:' separated list PATH until one succeeds in a successful lookup.
   If none succeed, then the first error that wasn't ENOENT is returned, or
   ENOENT if no other errors were returned.  If PREFIXED_NAME is non-NULL,
   then if RESULT is looked up directly, *PREFIXED_NAME is set to NULL, and
   if it is looked up using a prefix from PATH, *PREFIXED_NAME is set to
   malloced storage containing the prefixed name.  */
error_t
__hurd_file_name_path_lookup (error_t (*use_init_port)
			        (int which, error_t (*operate) (mach_port_t)),
			      file_t (*get_dtable_port) (int fd),
			      error_t (*lookup)
			        (file_t dir, const char *name, int flags, mode_t mode,
			         retry_type *do_retry, string_t retry_name,
			         mach_port_t *result),
			      const char *file_name, const char *path,
			      int flags, mode_t mode,
			      file_t *result, char **prefixed_name)
{
  error_t scan_lookup (const char *name)
    {
      return
	__hurd_file_name_lookup (use_init_port, get_dtable_port, lookup,
				 name, flags, mode, result);
    }
  return file_name_path_scan (file_name, path, scan_lookup, prefixed_name);
}
strong_alias (__hurd_file_name_path_lookup, hurd_file_name_path_lookup)

file_t
file_name_path_lookup (const char *file_name, const char *path,
		       int flags, mode_t mode, char **prefixed_name)
{
  error_t err;
  file_t result;

  err = __hurd_file_name_path_lookup (&_hurd_ports_use, &__getdport, 0,
				      file_name, path, flags, mode,
				      &result, prefixed_name);

  return err ? (__hurd_fail (err), MACH_PORT_NULL) : result;
}
