/* Declarations of file name translation functions for the GNU Hurd.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
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

#ifndef _HURD_LOOKUP_H
#define _HURD_LOOKUP_H	1

#include <errno.h>
#include <bits/types/error_t.h>
#include <hurd/hurd_types.h>

/* These functions all take two callback functions as the first two arguments.
   The first callback function USE_INIT_PORT is called as follows:

   error_t use_init_port (int which, error_t (*operate) (mach_port_t));

   WHICH is nonnegative value less than INIT_PORT_MAX, indicating which
   init port is required.  The callback function should call *OPERATE
   with a send right to the appropriate init port.  No user reference
   is consumed; the right will only be used after *OPERATE returns if
   *OPERATE has added its own user reference.

   LOOKUP is a function to do the actual filesystem lookup.  It is passed the
   same arguments that the dir_lookup rpc accepts, and if 0, __dir_lookup is
   used.

   The second callback function GET_DTABLE_PORT should behave like `getdport'.

   All these functions return zero on success or an error code on failure.  */


/* Open a port to FILE with the given FLAGS and MODE (see <fcntl.h>).  If
   successful, returns zero and store the port to FILE in *PORT; otherwise
   returns an error code. */

error_t __hurd_file_name_lookup (error_t (*use_init_port)
				   (int which,
				    error_t (*operate) (mach_port_t)),
				 file_t (*get_dtable_port) (int fd),
				 error_t (*lookup)
				   (file_t dir, const char *name, int flags, mode_t mode,
				    retry_type *do_retry, string_t retry_name,
				    mach_port_t *result),
				 const char *file_name,
				 int flags, mode_t mode,
				 file_t *result);
error_t hurd_file_name_lookup (error_t (*use_init_port)
			         (int which,
				  error_t (*operate) (mach_port_t)),
			       file_t (*get_dtable_port) (int fd),
			       error_t (*lookup)
				 (file_t dir, const char *name, int flags, mode_t mode,
				  retry_type *do_retry, string_t retry_name,
				  mach_port_t *result),
			       const char *file_name,
			       int flags, mode_t mode,
			       file_t *result);


/* Split FILE into a directory and a name within the directory.  Look up a
   port for the directory and store it in *DIR; store in *NAME a pointer
   into FILE where the name within directory begins.  */

error_t __hurd_file_name_split (error_t (*use_init_port)
				  (int which,
				   error_t (*operate) (mach_port_t)),
				file_t (*get_dtable_port) (int fd),
				error_t (*lookup) (file_t dir, const char *name,
						   int flags, mode_t mode,
				   retry_type *do_retry, string_t retry_name,
				   mach_port_t *result),
				const char *file_name,
				file_t *dir, char **name);
error_t hurd_file_name_split (error_t (*use_init_port)
			        (int which,
				 error_t (*operate) (mach_port_t)),
			      file_t (*get_dtable_port) (int fd),
			      error_t (*lookup) (file_t dir, const char *name,
						 int flags, mode_t mode,
				 retry_type *do_retry, string_t retry_name,
				 mach_port_t *result),
			      const char *file_name,
			      file_t *dir, char **name);

/* Split DIRECTORY into a parent directory and a name within the directory.
   This is the same as hurd_file_name_split, but ignores trailing slashes.  */

error_t __hurd_directory_name_split (error_t (*use_init_port)
				  (int which,
				   error_t (*operate) (mach_port_t)),
				file_t (*get_dtable_port) (int fd),
				error_t (*lookup) (file_t dir, const char *name,
						   int flags, mode_t mode,
				   retry_type *do_retry, string_t retry_name,
				   mach_port_t *result),
				const char *directory_name,
				file_t *dir, char **name);
error_t hurd_directory_name_split (error_t (*use_init_port)
				   (int which,
				    error_t (*operate) (mach_port_t)),
				   file_t (*get_dtable_port) (int fd),
				   error_t (*lookup) (file_t dir, const char *name,
						      int flags, mode_t mode,
				    retry_type *do_retry, string_t retry_name,
				    mach_port_t *result),
				   const char *directory_name,
				   file_t *dir, char **name);


/* Process the values returned by `dir_lookup' et al, and loop doing
   `dir_lookup' calls until one returns FS_RETRY_NONE.  The arguments
   should be those just passed to and/or returned from `dir_lookup',
   `fsys_getroot', or `file_invoke_translator'.  This function consumes the
   reference in *RESULT even if it returns an error.  */

error_t __hurd_file_name_lookup_retry (error_t (*use_init_port)
				         (int which,
					  error_t (*operate) (mach_port_t)),
				       file_t (*get_dtable_port) (int fd),
				       error_t (*lookup)
				         (file_t dir, const char *name,
					  int flags, mode_t mode,
					  retry_type *do_retry,
					  string_t retry_name,
					  mach_port_t *result),
				       enum retry_type doretry,
				       char retryname[1024],
				       int flags, mode_t mode,
				       file_t *result);
error_t hurd_file_name_lookup_retry (error_t (*use_init_port)
				       (int which,
					error_t (*operate) (mach_port_t)),
				     file_t (*get_dtable_port) (int fd),
				     error_t (*lookup)
				       (file_t dir, const char *name,
					int flags, mode_t mode,
					retry_type *do_retry,
					string_t retry_name,
					mach_port_t *result),
				     enum retry_type doretry,
				     char retryname[1024],
				     int flags, mode_t mode,
				     file_t *result);


/* If FILE_NAME contains a '/', or PATH is NULL, call FUN with FILE_NAME, and
   return the result (if PREFIXED_NAME is non-NULL, setting *PREFIXED_NAME to
   NULL).  Otherwise, call FUN repeatedly with FILE_NAME prefixed with each
   successive `:' separated element of PATH, returning whenever FUN returns
   0 (if PREFIXED_NAME is non-NULL, setting *PREFIXED_NAME to the resulting
   prefixed path).  If FUN never returns 0, return the first non-ENOENT
   return value, or ENOENT if there is none.  */
error_t file_name_path_scan (const char *file_name, const char *path,
			     error_t (*fun)(const char *name),
			     char **prefixed_name);

/* Lookup FILE_NAME and return the node opened with FLAGS & MODE in result
   (see hurd_file_name_lookup for details), but a simple filename (without
   any directory prefixes) will be consecutively prefixed with the pathnames
   in the `:' separated list PATH until one succeeds in a successful lookup.
   If none succeed, then the first error that wasn't ENOENT is returned, or
   ENOENT if no other errors were returned.  If PREFIXED_NAME is non-NULL,
   then if RESULT is looked up directly, *PREFIXED_NAME is set to NULL, and
   if it is looked up using a prefix from PATH, *PREFIXED_NAME is set to
   malloced storage containing the prefixed name.  */
error_t hurd_file_name_path_lookup (error_t (*use_init_port)
				    (int which,
				     error_t (*operate) (mach_port_t)),
				    file_t (*get_dtable_port) (int fd),
				    error_t (*lookup)
				      (file_t dir, const char *name,
				       int flags, mode_t mode,
				       retry_type *do_retry,
				       string_t retry_name,
				       mach_port_t *result),
				    const char *file_name, const char *path,
				    int flags, mode_t mode,
				    file_t *result, char **prefixed_name);

#endif	/* hurd/lookup.h */
