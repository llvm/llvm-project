/* Copyright (C) 1992-2021 Free Software Foundation, Inc.
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

#include <hurd.h>
#include <hurd/lookup.h>
#include <string.h>
#include <fcntl.h>


/* Translate the error from dir_lookup into the error the user sees.  */
static inline error_t
lookup_error (error_t error)
{
  switch (error)
    {
    case EOPNOTSUPP:
    case MIG_BAD_ID:
      /* These indicate that the server does not understand dir_lookup
	 at all.  If it were a directory, it would, by definition.  */
      return ENOTDIR;
    default:
      return error;
    }
}

error_t
__hurd_file_name_lookup (error_t (*use_init_port)
			   (int which, error_t (*operate) (file_t)),
			 file_t (*get_dtable_port) (int fd),
			 error_t (*lookup)
			   (file_t dir, const char *name, int flags, mode_t mode,
			    retry_type *do_retry, string_t retry_name,
			    mach_port_t *result),
			 const char *file_name, int flags, mode_t mode,
			 file_t *result)
{
  error_t err;
  enum retry_type doretry;
  char retryname[1024];		/* XXX string_t LOSES! */
  int startport;

  error_t lookup_op (mach_port_t startdir)
    {
      return lookup_error ((*lookup) (startdir, file_name, flags, mode,
				      &doretry, retryname, result));
    }

  if (! lookup)
    lookup = __dir_lookup;

  if (file_name[0] == '\0')
    return ENOENT;

  startport = (file_name[0] == '/') ? INIT_PORT_CRDIR : INIT_PORT_CWDIR;
  while (file_name[0] == '/')
    file_name++;

  if (flags & O_NOFOLLOW)	/* See lookup-retry.c about O_NOFOLLOW.  */
    flags |= O_NOTRANS;

  if (flags & O_DIRECTORY && (flags & O_NOFOLLOW) == 0)
    {
      /* The caller wants to require that the file we look up is a directory.
	 We can do this without an extra RPC by appending a trailing slash
	 to the file name we look up.  */
      size_t len = strlen (file_name);
      if (len == 0)
	file_name = "/";
      else if (file_name[len - 1] != '/')
	{
	  char *n = alloca (len + 2);
	  memcpy (n, file_name, len);
	  n[len] = '/';
	  n[len + 1] = '\0';
	  file_name = n;
	}
    }

  err = (*use_init_port) (startport, &lookup_op);
  if (! err)
    err = __hurd_file_name_lookup_retry (use_init_port, get_dtable_port,
					 lookup, doretry, retryname,
					 flags, mode, result);

  return err;
}
weak_alias (__hurd_file_name_lookup, hurd_file_name_lookup)

error_t
__hurd_file_name_split (error_t (*use_init_port)
			  (int which, error_t (*operate) (file_t)),
			file_t (*get_dtable_port) (int fd),
			error_t (*lookup)
			  (file_t dir, const char *name, int flags, mode_t mode,
			   retry_type *do_retry, string_t retry_name,
			   mach_port_t *result),
			const char *file_name,
			file_t *dir, char **name)
{
  error_t addref (file_t crdir)
    {
      *dir = crdir;
      return __mach_port_mod_refs (__mach_task_self (),
				   crdir, MACH_PORT_RIGHT_SEND, +1);
    }

  const char *lastslash = strrchr (file_name, '/');

  if (lastslash != NULL)
    {
      if (lastslash == file_name)
	{
	  /* "/foobar" => crdir + "foobar".  */
	  *name = (char *) file_name + 1;
	  return (*use_init_port) (INIT_PORT_CRDIR, &addref);
	}
      else
	{
	  /* "/dir1/dir2/.../file".  */
	  char dirname[lastslash - file_name + 1];
	  memcpy (dirname, file_name, lastslash - file_name);
	  dirname[lastslash - file_name] = '\0';
	  *name = (char *) lastslash + 1;
	  return
	    __hurd_file_name_lookup (use_init_port, get_dtable_port, lookup,
				     dirname, 0, 0, dir);
	}
    }
  else if (file_name[0] == '\0')
    return ENOENT;
  else
    {
      /* "foobar" => cwdir + "foobar".  */
      *name = (char *) file_name;
      return (*use_init_port) (INIT_PORT_CWDIR, &addref);
    }
}
weak_alias (__hurd_file_name_split, hurd_file_name_split)

/* This is the same as hurd_file_name_split, except that it ignores
   trailing slashes (so *NAME is never "").  */
error_t
__hurd_directory_name_split (error_t (*use_init_port)
			     (int which, error_t (*operate) (file_t)),
			     file_t (*get_dtable_port) (int fd),
			     error_t (*lookup)
			     (file_t dir, const char *name, int flags, mode_t mode,
			      retry_type *do_retry, string_t retry_name,
			      mach_port_t *result),
			     const char *file_name,
			     file_t *dir, char **name)
{
  error_t addref (file_t crdir)
    {
      *dir = crdir;
      return __mach_port_mod_refs (__mach_task_self (),
				   crdir, MACH_PORT_RIGHT_SEND, +1);
    }

  const char *lastslash = strrchr (file_name, '/');

  if (lastslash != NULL && lastslash[1] == '\0')
    {
      /* Trailing slash doesn't count.  Look back further.  */

      /* Back up over all trailing slashes.  */
      while (lastslash > file_name && *lastslash == '/')
	--lastslash;

      /* Find the last one earlier in the string, before the trailing ones.  */
      lastslash = __memrchr (file_name, '/', lastslash - file_name);
    }

  if (lastslash != NULL)
    {
      if (lastslash == file_name)
	{
	  /* "/foobar" => crdir + "foobar".  */
	  *name = (char *) file_name + 1;
	  return (*use_init_port) (INIT_PORT_CRDIR, &addref);
	}
      else
	{
	  /* "/dir1/dir2/.../file".  */
	  char dirname[lastslash - file_name + 1];
	  memcpy (dirname, file_name, lastslash - file_name);
	  dirname[lastslash - file_name] = '\0';
	  *name = (char *) lastslash + 1;
	  return
	    __hurd_file_name_lookup (use_init_port, get_dtable_port, lookup,
				     dirname, 0, 0, dir);
	}
    }
  else if (file_name[0] == '\0')
    return ENOENT;
  else
    {
      /* "foobar" => cwdir + "foobar".  */
      *name = (char *) file_name;
      return (*use_init_port) (INIT_PORT_CWDIR, &addref);
    }
}
weak_alias (__hurd_directory_name_split, hurd_directory_name_split)


file_t
__file_name_lookup (const char *file_name, int flags, mode_t mode)
{
  error_t err;
  file_t result;

  err = __hurd_file_name_lookup (&_hurd_ports_use, &__getdport, 0,
				 file_name, flags, mode & ~_hurd_umask,
				 &result);

  return err ? (__hurd_fail (err), MACH_PORT_NULL) : result;
}
weak_alias (__file_name_lookup, file_name_lookup)


file_t
__file_name_split (const char *file_name, char **name)
{
  error_t err;
  file_t result;

  err = __hurd_file_name_split (&_hurd_ports_use, &__getdport, 0,
				file_name, &result, name);

  return err ? (__hurd_fail (err), MACH_PORT_NULL) : result;
}
weak_alias (__file_name_split, file_name_split)

file_t
__directory_name_split (const char *directory_name, char **name)
{
  error_t err;
  file_t result;

  err = __hurd_directory_name_split (&_hurd_ports_use, &__getdport, 0,
				     directory_name, &result, name);

  return err ? (__hurd_fail (err), MACH_PORT_NULL) : result;
}
weak_alias (__directory_name_split, directory_name_split)


file_t
__file_name_lookup_under (file_t startdir,
			  const char *file_name, int flags, mode_t mode)
{
  error_t err;
  file_t result;

  error_t use_init_port (int which, error_t (*operate) (mach_port_t))
    {
      return (which == INIT_PORT_CWDIR ? (*operate) (startdir)
	      : _hurd_ports_use (which, operate));
    }

  err = __hurd_file_name_lookup (&use_init_port, &__getdport, 0,
				 file_name, flags, mode & ~_hurd_umask,
				 &result);

  return err ? (__hurd_fail (err), MACH_PORT_NULL) : result;
}
weak_alias (__file_name_lookup_under, file_name_lookup_under)
