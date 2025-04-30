/* Support for *xattr interfaces on GNU/Hurd.
   Copyright (C) 2006-2021 Free Software Foundation, Inc.
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
#include <hurd/xattr.h>
#include <string.h>
#include <sys/mman.h>

/* Right now we support only a fixed set of xattr names for Hurd features.
   There are no RPC interfaces for free-form xattr names and values.

   Name			Value encoding
   ----			----- --------
   gnu.author		empty if st_author==st_uid
			uid_t giving st_author value
   gnu.translator	empty if no passive translator
			translator and arguments: "/hurd/foo\0arg1\0arg2\0"
*/

error_t
_hurd_xattr_get (io_t port, const char *name, void *value, size_t *size)
{
  if (strncmp (name, "gnu.", 4))
    return EOPNOTSUPP;
  name += 4;

  if (!strcmp (name, "author"))
    {
      struct stat64 st;
      error_t err = __io_stat (port, &st);
      if (err)
	return err;
      if (st.st_author == st.st_uid)
	*size = 0;
      else if (value)
	{
	  if (*size < sizeof st.st_author)
	    return ERANGE;
	  memcpy (value, &st.st_author, sizeof st.st_author);
	}
      *size = sizeof st.st_author;
      return 0;
    }

  if (!strcmp (name, "translator"))
    {
      char *buf = value;
      size_t bufsz = value ? *size : 0;
      error_t err = __file_get_translator (port, &buf, &bufsz);
      if (err)
	return err;
      if (value != NULL && *size < bufsz)
	{
	  if (buf != value)
	    __munmap (buf, bufsz);
	  return -ERANGE;
	}
      if (buf != value && bufsz > 0)
	{
	  if (value != NULL)
	    memcpy (value, buf, bufsz);
	  __munmap (buf, bufsz);
	}
      *size = bufsz;
      return 0;
    }

  return EOPNOTSUPP;
}

error_t
_hurd_xattr_set (io_t port, const char *name, const void *value, size_t size,
		 int flags)
{
  if (strncmp (name, "gnu.", 4))
    return EOPNOTSUPP;
  name += 4;

  if (!strcmp (name, "author"))
    switch (size)
      {
      default:
	return EINVAL;
      case 0:			/* "Clear" author by setting to st_uid. */
	{
	  struct stat64 st;
	  error_t err = __io_stat (port, &st);
	  if (err)
	    return err;
	  if (st.st_author == st.st_uid)
	    {
	      /* Nothing to do.  */
	      if (flags & XATTR_REPLACE)
		return ENODATA;
	      return 0;
	    }
	  if (flags & XATTR_CREATE)
	    return EEXIST;
	  return __file_chauthor (port, st.st_uid);
	}
      case sizeof (uid_t):	/* Set the author.  */
	{
	  uid_t id;
	  memcpy (&id, value, sizeof id);
	  if (flags & (XATTR_CREATE|XATTR_REPLACE))
	    {
	      struct stat64 st;
	      error_t err = __io_stat (port, &st);
	      if (err)
		return err;
	      if (st.st_author == st.st_uid)
		{
		  if (flags & XATTR_REPLACE)
		    return ENODATA;
		}
	      else if (flags & XATTR_CREATE)
		return EEXIST;
	      if (st.st_author == id)
		/* Nothing to do.  */
		return 0;
	    }
	  return __file_chauthor (port, id);
	}
      }

  if (!strcmp (name, "translator"))
    {
      if (flags & XATTR_REPLACE)
	{
	  /* Must make sure it's already there.  */
	  char *buf = NULL;
	  size_t bufsz = 0;
	  error_t err = __file_get_translator (port, &buf, &bufsz);
	  if (err)
	    return err;
	  if (bufsz > 0)
	    {
	      __munmap (buf, bufsz);
	      return ENODATA;
	    }
	}
      return __file_set_translator (port,
				    FS_TRANS_SET | ((flags & XATTR_CREATE)
						    ? FS_TRANS_EXCL : 0), 0, 0,
				    value, size,
				    MACH_PORT_NULL, MACH_MSG_TYPE_COPY_SEND);
    }

  return EOPNOTSUPP;
}

error_t
_hurd_xattr_remove (io_t port, const char *name)
{
  return _hurd_xattr_set (port, name, NULL, 0, XATTR_REPLACE);
}

error_t
_hurd_xattr_list (io_t port, void *buffer, size_t *size)
{
  size_t total = 0;
  char *bufp = buffer;
  inline void add (const char *name, size_t len)
    {
      total += len;
      if (bufp != NULL && total <= *size)
	bufp = __mempcpy (bufp, name, len);
    }
#define add(s) add (s, sizeof s)

  struct stat64 st;
  error_t err = __io_stat (port, &st);
  if (err)
    return err;

  if (st.st_author != st.st_uid)
    add ("gnu.author");
  if (st.st_mode & S_IPTRANS)
    add ("gnu.translator");

  if (buffer != NULL && total > *size)
    return ERANGE;
  *size = total;
  return 0;
}
