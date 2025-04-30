/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>
   and Paul Janzen <pcj@primenet.com>, 1996.

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

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <utmp.h>
#include <not-cancel.h>
#include <kernel-features.h>
#include <sigsetops.h>
#include <not-cancel.h>

#include "utmp-private.h"
#include "utmp-equal.h"


/* Descriptor for the file and position.  */
static int file_fd = -1;
static bool file_writable;
static off64_t file_offset;

/* Cache for the last read entry.  */
static struct utmp last_entry;

/* Returns true if *ENTRY matches last_entry, based on
   data->ut_type.  */
static bool
matches_last_entry (const struct utmp *data)
{
  if (file_offset <= 0)
    /* Nothing has been read.  last_entry is stale and cannot match.  */
    return false;

  if (data->ut_type == RUN_LVL
      || data->ut_type == BOOT_TIME
      || data->ut_type == OLD_TIME
      || data->ut_type == NEW_TIME)
    /* For some entry types, only a type match is required.  */
    return data->ut_type == last_entry.ut_type;
  else
    /* For the process-related entries, a full match is needed.  */
    return __utmp_equal (&last_entry, data);
}

/* Locking timeout.  */
#ifndef TIMEOUT
# define TIMEOUT 10
#endif

/* Do-nothing handler for locking timeout.  */
static void timeout_handler (int signum) {};


/* try_file_lock (LOCKING, FD, TYPE) returns true if the locking
   operation failed and recovery needs to be performed.

   file_unlock (FD) removes the lock (which must have been
   successfully acquired). */

static bool
try_file_lock (int fd, int type)
{
  /* Cancel any existing alarm.  */
  int old_timeout = alarm (0);

  /* Establish signal handler.  */
  struct sigaction old_action;
  struct sigaction action;
  action.sa_handler = timeout_handler;
  __sigemptyset (&action.sa_mask);
  action.sa_flags = 0;
  __sigaction (SIGALRM, &action, &old_action);

  alarm (TIMEOUT);

  /* Try to get the lock.  */
 struct flock64 fl =
   {
    .l_type = type,
    .l_whence = SEEK_SET,
   };

 bool status = __fcntl64_nocancel (fd, F_SETLKW, &fl) < 0;
 int saved_errno = errno;

 /* Reset the signal handler and alarm.  We must reset the alarm
    before resetting the handler so our alarm does not generate a
    spurious SIGALRM seen by the user.  However, we cannot just set
    the user's old alarm before restoring the handler, because then
    it's possible our handler could catch the user alarm's SIGARLM and
    then the user would never see the signal he expected.  */
  alarm (0);
  __sigaction (SIGALRM, &old_action, NULL);
  if (old_timeout != 0)
    alarm (old_timeout);

  __set_errno (saved_errno);
  return status;
}

static void
file_unlock (int fd)
{
  struct flock64 fl =
    {
      .l_type = F_UNLCK,
    };
  __fcntl64_nocancel (fd, F_SETLKW, &fl);
}

#ifndef TRANSFORM_UTMP_FILE_NAME
# define TRANSFORM_UTMP_FILE_NAME(file_name) (file_name)
#endif

int
__libc_setutent (void)
{
  if (file_fd < 0)
    {
      const char *file_name;

      file_name = TRANSFORM_UTMP_FILE_NAME (__libc_utmp_file_name);

      file_writable = false;
      file_fd = __open_nocancel
	(file_name, O_RDONLY | O_LARGEFILE | O_CLOEXEC);
      if (file_fd == -1)
	return 0;
    }

  __lseek64 (file_fd, 0, SEEK_SET);
  file_offset = 0;

  return 1;
}

/* Preform initialization if necessary.  */
static bool
maybe_setutent (void)
{
  return file_fd >= 0 || __libc_setutent ();
}

/* Reads the entry at file_offset, storing it in last_entry and
   updating file_offset on success.  Returns -1 for a read error, 0
   for EOF, and 1 for a successful read.  last_entry and file_offset
   are only updated on a successful and complete read.  */
static ssize_t
read_last_entry (void)
{
  struct utmp buffer;
  ssize_t nbytes = __pread64_nocancel (file_fd, &buffer, sizeof (buffer),
				       file_offset);
  if (nbytes < 0)
    return -1;
  else if (nbytes != sizeof (buffer))
    /* Assume EOF.  */
    return 0;
  else
    {
      last_entry = buffer;
      file_offset += sizeof (buffer);
      return 1;
    }
}

int
__libc_getutent_r (struct utmp *buffer, struct utmp **result)
{
  int saved_errno = errno;

  if (!maybe_setutent ())
    {
      /* Not available.  */
      *result = NULL;
      return -1;
    }

  if (try_file_lock (file_fd, F_RDLCK))
    return -1;

  ssize_t nbytes = read_last_entry ();
  file_unlock (file_fd);

  if (nbytes <= 0)		/* Read error or EOF.  */
    {
      if (nbytes == 0)
	/* errno should be unchanged to indicate success.  A premature
	   EOF is treated like an EOF (missing complete record at the
	   end).  */
	__set_errno (saved_errno);
      *result = NULL;
      return -1;
    }

  memcpy (buffer, &last_entry, sizeof (struct utmp));
  *result = buffer;

  return 0;
}


/* Search for *ID, updating last_entry and file_offset.  Return 0 on
   success and -1 on failure.  Does not perform locking; for that see
   internal_getut_r below.  */
static int
internal_getut_nolock (const struct utmp *id)
{
  while (1)
    {
      ssize_t nbytes = read_last_entry ();
      if (nbytes < 0)
	return -1;
      if (nbytes == 0)
	{
	  /* End of file reached.  */
	  __set_errno (ESRCH);
	  return -1;
	}

      if (matches_last_entry (id))
	break;
    }

  return 0;
}

/* Search for *ID, updating last_entry and file_offset.  Return 0 on
   success and -1 on failure.  If the locking operation failed, write
   true to *LOCK_FAILED.  */
static int
internal_getut_r (const struct utmp *id, bool *lock_failed)
{
  if (try_file_lock (file_fd, F_RDLCK))
    {
      *lock_failed = true;
      return -1;
    }

  int result = internal_getut_nolock (id);
  file_unlock (file_fd);
  return result;
}

/* For implementing this function we don't use the getutent_r function
   because we can avoid the reposition on every new entry this way.  */
int
__libc_getutid_r (const struct utmp *id, struct utmp *buffer,
		  struct utmp **result)
{
  if (!maybe_setutent ())
    {
      *result = NULL;
      return -1;
    }

  /* We don't have to distinguish whether we can lock the file or
     whether there is no entry.  */
  bool lock_failed = false;
  if (internal_getut_r (id, &lock_failed) < 0)
    {
      *result = NULL;
      return -1;
    }

  memcpy (buffer, &last_entry, sizeof (struct utmp));
  *result = buffer;

  return 0;
}

/* For implementing this function we don't use the getutent_r function
   because we can avoid the reposition on every new entry this way.  */
int
__libc_getutline_r (const struct utmp *line, struct utmp *buffer,
		    struct utmp **result)
{
  if (!maybe_setutent ())
    {
      *result = NULL;
      return -1;
    }

  if (try_file_lock (file_fd, F_RDLCK))
    {
      *result = NULL;
      return -1;
    }

  while (1)
    {
      ssize_t nbytes = read_last_entry ();
      if (nbytes < 0)
	{
	  file_unlock (file_fd);
	  *result = NULL;
	  return -1;
	}
      if (nbytes == 0)
	{
	  /* End of file reached.  */
	  file_unlock (file_fd);
	  __set_errno (ESRCH);
	  *result = NULL;
	  return -1;
	}

      /* Stop if we found a user or login entry.  */
      if ((last_entry.ut_type == USER_PROCESS
	   || last_entry.ut_type == LOGIN_PROCESS)
	  && (strncmp (line->ut_line, last_entry.ut_line, sizeof line->ut_line)
	      == 0))
	break;
    }

  file_unlock (file_fd);
  memcpy (buffer, &last_entry, sizeof (struct utmp));
  *result = buffer;

  return 0;
}


struct utmp *
__libc_pututline (const struct utmp *data)
{
  if (!maybe_setutent ())
    return NULL;

  struct utmp *pbuf;

  if (! file_writable)
    {
      /* We must make the file descriptor writable before going on.  */
      const char *file_name = TRANSFORM_UTMP_FILE_NAME (__libc_utmp_file_name);

      int new_fd = __open_nocancel
	(file_name, O_RDWR | O_LARGEFILE | O_CLOEXEC);
      if (new_fd == -1)
	return NULL;

      if (__dup2 (new_fd, file_fd) < 0)
	{
	  __close_nocancel_nostatus (new_fd);
	  return NULL;
	}
      __close_nocancel_nostatus (new_fd);
      file_writable = true;
    }

  /* Exclude other writers before validating the cache.  */
  if (try_file_lock (file_fd, F_WRLCK))
    return NULL;

  /* Find the correct place to insert the data.  */
  bool found = false;
  if (matches_last_entry (data))
    {
      /* Read back the entry under the write lock.  */
      file_offset -= sizeof (last_entry);
      ssize_t nbytes = read_last_entry ();
      if (nbytes < 0)
	{
	  file_unlock (file_fd);
	  return NULL;
	}

      if (nbytes == 0)
	/* End of file reached.  */
	found = false;
      else
	found = matches_last_entry (data);
    }

  if (!found)
    /* Search forward for the entry.  */
    found = internal_getut_nolock (data) >= 0;

  off64_t write_offset;
  if (!found)
    {
      /* We append the next entry.  */
      write_offset = __lseek64 (file_fd, 0, SEEK_END);

      /* Round down to the next multiple of the entry size.  This
	 ensures any partially-written record is overwritten by the
	 new record.  */
      write_offset = (write_offset / sizeof (struct utmp)
		      * sizeof (struct utmp));
    }
  else
    /* Overwrite last_entry.  */
    write_offset = file_offset - sizeof (struct utmp);

  /* Write the new data.  */
  ssize_t nbytes;
  if (__lseek64 (file_fd, write_offset, SEEK_SET) < 0
      || (nbytes = __write_nocancel (file_fd, data, sizeof (struct utmp))) < 0)
    {
      /* There is no need to recover the file position because all
	 reads use pread64, and any future write is preceded by
	 another seek.  */
      file_unlock (file_fd);
      return NULL;
    }

  if (nbytes != sizeof (struct utmp))
    {
      /* If we appended a new record this is only partially written.
	 Remove it.  */
      if (!found)
	(void) __ftruncate64 (file_fd, write_offset);
      file_unlock (file_fd);
      /* Assume that the write failure was due to missing disk
	 space.  */
      __set_errno (ENOSPC);
      return NULL;
    }

  file_unlock (file_fd);
  file_offset = write_offset + sizeof (struct utmp);
  pbuf = (struct utmp *) data;

  return pbuf;
}


void
__libc_endutent (void)
{
  if (file_fd >= 0)
    {
      __close_nocancel_nostatus (file_fd);
      file_fd = -1;
    }
}


int
__libc_updwtmp (const char *file, const struct utmp *utmp)
{
  int result = -1;
  off64_t offset;
  int fd;

  /* Open WTMP file.  */
  fd = __open_nocancel (file, O_WRONLY | O_LARGEFILE);
  if (fd < 0)
    return -1;

  if (try_file_lock (fd, F_WRLCK))
    {
      __close_nocancel_nostatus (fd);
      return -1;
    }

  /* Remember original size of log file.  */
  offset = __lseek64 (fd, 0, SEEK_END);
  if (offset % sizeof (struct utmp) != 0)
    {
      offset -= offset % sizeof (struct utmp);
      __ftruncate64 (fd, offset);

      if (__lseek64 (fd, 0, SEEK_END) < 0)
	goto unlock_return;
    }

  /* Write the entry.  If we can't write all the bytes, reset the file
     size back to the original size.  That way, no partial entries
     will remain.  */
  if (__write_nocancel (fd, utmp, sizeof (struct utmp))
      != sizeof (struct utmp))
    {
      __ftruncate64 (fd, offset);
      goto unlock_return;
    }

  result = 0;

unlock_return:
  file_unlock (fd);

  /* Close WTMP file.  */
  __close_nocancel_nostatus (fd);

  return result;
}
