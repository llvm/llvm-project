/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <hurd.h>
#include <hurd/port.h>
#include <dirent.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>


/* Get the canonical absolute name of the given directory port, and put it
   in SIZE bytes of BUF.  Returns NULL if the directory couldn't be
   determined or SIZE was too small.  If successful, returns BUF.  In GNU,
   if BUF is NULL, an array is allocated with `malloc'; the array is SIZE
   bytes long, unless SIZE <= 0, in which case it is as big as necessary.
   If our root directory cannot be reached, the result will not begin with
   a slash to indicate that it is relative to some unknown root directory.  */

char *
__hurd_canonicalize_directory_name_internal (file_t thisdir,
					    char *buf,
					    size_t size)
{
  error_t err;
  mach_port_t rootid, thisid, rootdevid, thisdevid;
  ino64_t rootino, thisino;
  char *file_name;
  char *file_namep;
  file_t parent;
  char *dirbuf = NULL;
  unsigned int dirbufsize = 0;
  const size_t orig_size = size;

  inline void cleanup (void)
    {
      if (parent != thisdir)
	__mach_port_deallocate (__mach_task_self (), parent);

      __mach_port_deallocate (__mach_task_self (), thisid);
      __mach_port_deallocate (__mach_task_self (), thisdevid);
      __mach_port_deallocate (__mach_task_self (), rootid);

      if (dirbuf != NULL)
	__vm_deallocate (__mach_task_self (),
			 (vm_address_t) dirbuf, dirbufsize);
    }


  if (size <= 0)
    {
      if (buf != NULL)
	{
	  errno = EINVAL;
	  return NULL;
	}

      size = FILENAME_MAX * 4 + 1;	/* Good starting guess.  */
    }

  if (buf != NULL)
    file_name = buf;
  else
    {
      file_name = malloc (size);
      if (file_name == NULL)
	return NULL;
    }

  file_namep = file_name + size;
  *--file_namep = '\0';

  /* Get a port to our root directory and get its identity.  */

  if (err = __USEPORT (CRDIR, __io_identity (port,
					     &rootid, &rootdevid, &rootino)))
    return __hurd_fail (err), NULL;
  __mach_port_deallocate (__mach_task_self (), rootdevid);

  /* Stat the port to the directory of interest.  */

  if (err = __io_identity (thisdir, &thisid, &thisdevid, &thisino))
    {
      __mach_port_deallocate (__mach_task_self (), rootid);
      return __hurd_fail (err), NULL;
    }

  parent = thisdir;
  while (thisid != rootid)
    {
      /* PARENT is a port to the directory we are currently on;
	 THISID, THISDEV, and THISINO are its identity.
	 Look in its parent (..) for a file with the same file number.  */

      struct dirent64 *d;
      mach_port_t dotid, dotdevid;
      ino64_t dotino;
      int mount_point;
      file_t newp;
      char *dirdata;
      size_t dirdatasize;
      int direntry, nentries;


      /* Look at the parent directory.  */
      newp = __file_name_lookup_under (parent, "..", O_READ, 0);
      if (newp == MACH_PORT_NULL)
	goto lose;
      if (parent != thisdir)
	__mach_port_deallocate (__mach_task_self (), parent);
      parent = newp;

      /* Get this directory's identity and figure out if it's a mount
         point.  */
      if (err = __io_identity (parent, &dotid, &dotdevid, &dotino))
	goto errlose;
      mount_point = dotdevid != thisdevid;

      if (thisid == dotid)
	{
	  /* `..' == `.' but it is not our root directory.  */
	  __mach_port_deallocate (__mach_task_self (), dotid);
	  __mach_port_deallocate (__mach_task_self (), dotdevid);
	  break;
	}

      /* Search for the last directory.  */
      direntry = 0;
      dirdata = dirbuf;
      dirdatasize = dirbufsize;
      while (!(err = __dir_readdir (parent, &dirdata, &dirdatasize,
				    direntry, -1, 0, &nentries))
	     && nentries != 0)
	{
	  /* We have a block of directory entries.  */

	  unsigned int offset;

	  direntry += nentries;

	  if (dirdata != dirbuf)
	    {
	      /* The data was passed out of line, so our old buffer is no
		 longer useful.  Deallocate the old buffer and reset our
		 information for the new buffer.  */
	      __vm_deallocate (__mach_task_self (),
			       (vm_address_t) dirbuf, dirbufsize);
	      dirbuf = dirdata;
	      dirbufsize = round_page (dirdatasize);
	    }

	  /* Iterate over the returned directory entries, looking for one
	     whose file number is THISINO.  */

	  offset = 0;
	  while (offset < dirdatasize)
	    {
	      d = (struct dirent64 *) &dirdata[offset];
	      offset += d->d_reclen;

	      /* Ignore `.' and `..'.  */
	      if (d->d_name[0] == '.'
		  && (d->d_namlen == 1
		      || (d->d_namlen == 2 && d->d_name[1] == '.')))
		continue;

	      if (mount_point || d->d_ino == thisino)
		{
		  file_t try = __file_name_lookup_under (parent, d->d_name,
							 O_NOLINK, 0);
		  file_t id, devid;
		  ino64_t fileno;
		  if (try == MACH_PORT_NULL)
		    goto lose;
		  err = __io_identity (try, &id, &devid, &fileno);
		  __mach_port_deallocate (__mach_task_self (), try);
		  if (err)
		    goto inner_errlose;
		  __mach_port_deallocate (__mach_task_self (), id);
		  __mach_port_deallocate (__mach_task_self (), devid);
		  if (id == thisid)
		    goto found;
		}
	    }
	}

      if (err)
	{
	inner_errlose:		/* Goto ERRLOSE: after cleaning up.  */
	  __mach_port_deallocate (__mach_task_self (), dotid);
	  __mach_port_deallocate (__mach_task_self (), dotdevid);
	  goto errlose;
	}
      else if (nentries == 0)
	{
	  /* We got to the end of the directory without finding anything!
	     We are in a directory that has been unlinked, or something is
	     broken.  */
	  err = ENOENT;
	  goto inner_errlose;
	}
      else
      found:
	{
	  /* Prepend the directory name just discovered.  */

	  if (file_namep - file_name < d->d_namlen + 1)
	    {
	      if (orig_size > 0)
		{
		  errno = ERANGE;
		  return NULL;
		}
	      else
		{
		  size *= 2;
		  buf = realloc (file_name, size);
		  if (buf == NULL)
		    {
		      free (file_name);
		      return NULL;
		    }
		  file_namep = &buf[file_namep - file_name + size / 2];
		  file_name = buf;
		  /* Move current contents up to the end of the buffer.
		     This is guaranteed to be non-overlapping.  */
		  memcpy (file_namep, file_namep - size / 2,
			  file_name + size - file_namep);
		}
	    }
	  file_namep -= d->d_namlen;
	  (void) memcpy (file_namep, d->d_name, d->d_namlen);
	  *--file_namep = '/';
	}

      /* The next iteration will find the name of the directory we
	 just searched through.  */
      __mach_port_deallocate (__mach_task_self (), thisid);
      __mach_port_deallocate (__mach_task_self (), thisdevid);
      thisid = dotid;
      thisdevid = dotdevid;
      thisino = dotino;
    }

  if (file_namep == &file_name[size - 1])
    /* We found nothing and got all the way to the root.
       So the root is our current directory.  */
    *--file_namep = '/';

  memmove (file_name, file_namep, file_name + size - file_namep);
  cleanup ();
  return file_name;

 errlose:
  /* Set errno.  */
  (void) __hurd_fail (err);
 lose:
  cleanup ();
  return NULL;
}
strong_alias (__hurd_canonicalize_directory_name_internal, _hurd_canonicalize_directory_name_internal)

char *
__canonicalize_directory_name_internal (const char *thisdir, char *buf,
					size_t size)
{
  char *result;
  file_t port = __file_name_lookup (thisdir, 0, 0);
  if (port == MACH_PORT_NULL)
    return NULL;
  result = __hurd_canonicalize_directory_name_internal (port, buf, size);
  __mach_port_deallocate (__mach_task_self (), port);
  return result;
}

/* Get the pathname of the current working directory, and put it in SIZE
   bytes of BUF.  Returns NULL if the directory couldn't be determined or
   SIZE was too small.  If successful, returns BUF.  In GNU, if BUF is
   NULL, an array is allocated with `malloc'; the array is SIZE bytes long,
   unless SIZE <= 0, in which case it is as big as necessary.  */
char *
__getcwd (char *buf, size_t size)
{
  char *cwd =
    __USEPORT (CWDIR,
	       __hurd_canonicalize_directory_name_internal (port,
							    buf, size));
  return cwd;
}
libc_hidden_def (__getcwd)
weak_alias (__getcwd, getcwd)
