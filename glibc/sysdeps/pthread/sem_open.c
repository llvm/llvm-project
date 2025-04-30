/* Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2002.

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

#include <fcntl.h>
#include <semaphore.h>
#include <stdarg.h>
#include <unistd.h>
#include <sys/mman.h>
#include "semaphoreP.h"
#include <shm-directory.h>
#include <sem_routines.h>
#include <futex-internal.h>
#include <libc-lock.h>

#if !PTHREAD_IN_LIBC
/* The private names are not exported from libc.  */
# define __link link
# define __unlink unlink
#endif

sem_t *
__sem_open (const char *name, int oflag, ...)
{
  int fd;
  sem_t *result;

  /* Check that shared futexes are supported.  */
  int err = futex_supports_pshared (PTHREAD_PROCESS_SHARED);
  if (err != 0)
    {
      __set_errno (err);
      return SEM_FAILED;
    }

  struct shmdir_name dirname;
  if (__shm_get_name (&dirname, name, true) != 0)
    {
      __set_errno (EINVAL);
      return SEM_FAILED;
    }

  /* Disable asynchronous cancellation.  */
#ifdef __libc_ptf_call
  int state;
  __libc_ptf_call (__pthread_setcancelstate,
                   (PTHREAD_CANCEL_DISABLE, &state), 0);
#endif

  /* If the semaphore object has to exist simply open it.  */
  if ((oflag & O_CREAT) == 0 || (oflag & O_EXCL) == 0)
    {
    try_again:
      fd = __open (dirname.name,
		   (oflag & ~(O_CREAT|O_ACCMODE)) | O_NOFOLLOW | O_RDWR);

      if (fd == -1)
	{
	  /* If we are supposed to create the file try this next.  */
	  if ((oflag & O_CREAT) != 0 && errno == ENOENT)
	    goto try_create;

	  /* Return.  errno is already set.  */
	}
      else
	/* Check whether we already have this semaphore mapped and
	   create one if necessary.  */
	result = __sem_check_add_mapping (name, fd, SEM_FAILED);
    }
  else
    {
      /* We have to open a temporary file first since it must have the
	 correct form before we can start using it.  */
      mode_t mode;
      unsigned int value;
      va_list ap;

    try_create:
      va_start (ap, oflag);

      mode = va_arg (ap, mode_t);
      value = va_arg (ap, unsigned int);

      va_end (ap);

      if (value > SEM_VALUE_MAX)
	{
	  __set_errno (EINVAL);
	  result = SEM_FAILED;
	  goto out;
	}

      /* Create the initial file content.  */
      union
      {
	sem_t initsem;
	struct new_sem newsem;
      } sem;

      __new_sem_open_init (&sem.newsem, value);

      /* Initialize the remaining bytes as well.  */
      memset ((char *) &sem.initsem + sizeof (struct new_sem), '\0',
	      sizeof (sem_t) - sizeof (struct new_sem));

      char tmpfname[] = SHMDIR "sem.XXXXXX";
      int retries = 0;
#define NRETRIES 50
      while (1)
	{
	  /* We really want to use mktemp here.  We cannot use mkstemp
	     since the file must be opened with a specific mode.  The
	     mode cannot later be set since then we cannot apply the
	     file create mask.  */
	  if (__mktemp (tmpfname) == NULL)
	    {
	      result = SEM_FAILED;
	      goto out;
	    }

	  /* Open the file.  Make sure we do not overwrite anything.  */
	  fd = __open (tmpfname, O_RDWR | O_CREAT | O_EXCL, mode);
	  if (fd == -1)
	    {
	      if (errno == EEXIST)
		{
		  if (++retries < NRETRIES)
		    {
		      /* Restore the six placeholder bytes before the
			 null terminator before the next attempt.  */
		      memcpy (tmpfname + sizeof (tmpfname) - 7, "XXXXXX", 6);
		      continue;
		    }

		  __set_errno (EAGAIN);
		}

	      result = SEM_FAILED;
	      goto out;
	    }

	  /* We got a file.  */
	  break;
	}

      if (TEMP_FAILURE_RETRY (write (fd, &sem.initsem, sizeof (sem_t)))
	  == sizeof (sem_t)
	  /* Map the sem_t structure from the file.  */
	  && (result = (sem_t *) __mmap (NULL, sizeof (sem_t),
					 PROT_READ | PROT_WRITE, MAP_SHARED,
					 fd, 0)) != MAP_FAILED)
	{
	  /* Create the file.  Don't overwrite an existing file.  */
	  if (__link (tmpfname, dirname.name) != 0)
	    {
	      /* Undo the mapping.  */
	      __munmap (result, sizeof (sem_t));

	      /* Reinitialize 'result'.  */
	      result = SEM_FAILED;

	      /* This failed.  If O_EXCL is not set and the problem was
		 that the file exists, try again.  */
	      if ((oflag & O_EXCL) == 0 && errno == EEXIST)
		{
		  /* Remove the file.  */
		  __unlink (tmpfname);

		  /* Close the file.  */
		  __close (fd);

		  goto try_again;
		}
	    }
	  else
	    /* Insert the mapping into the search tree.  This also
	       determines whether another thread sneaked by and already
	       added such a mapping despite the fact that we created it.  */
	    result = __sem_check_add_mapping (name, fd, result);
	}

      /* Now remove the temporary name.  This should never fail.  If
	 it fails we leak a file name.  Better fix the kernel.  */
      __unlink (tmpfname);
    }

  /* Map the mmap error to the error we need.  */
  if (MAP_FAILED != (void *) SEM_FAILED && result == MAP_FAILED)
    result = SEM_FAILED;

  /* We don't need the file descriptor anymore.  */
  if (fd != -1)
    {
      /* Do not disturb errno.  */
      int save = errno;
      __close (fd);
      errno = save;
    }

out:
#ifdef __libc_ptf_call
  __libc_ptf_call (__pthread_setcancelstate, (state, NULL), 0);
#endif

  return result;
}
#if PTHREAD_IN_LIBC
versioned_symbol (libc, __sem_open, sem_open, GLIBC_2_34);
# if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_1_1, GLIBC_2_34)
compat_symbol (libpthread, __sem_open, sem_open, GLIBC_2_1_1);
# endif
#else /* !PTHREAD_IN_LIBC */
strong_alias (__sem_open, sem_open)
#endif
