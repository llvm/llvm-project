/* Handle locking of password file.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1996.

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
#include <libc-lock.h>
#include <shadow.h>
#include <signal.h>
#include <string.h>
#include <unistd.h>
#include <sys/file.h>
#include <sigsetops.h>

#include <kernel-features.h>


/* Name of the lock file.  */
#define PWD_LOCKFILE "/etc/.pwd.lock"

/* How long to wait for getting the lock before returning with an
   error.  */
#define TIMEOUT 15 /* sec */


/* File descriptor for lock file.  */
static int lock_fd = -1;

/* Prevent problems in multithreaded program by using mutex.  */
__libc_lock_define_initialized (static, lock)


/* Prototypes for local functions.  */
static void noop_handler (int __sig);


/* We cannot simply return in error cases.  We have to close the file
   and perhaps restore the signal handler.  */
#define RETURN_CLOSE_FD(code)						      \
  do {									      \
    if ((code) < 0 && lock_fd >= 0)					      \
      {									      \
	__close (lock_fd);						      \
	lock_fd = -1;							      \
      }									      \
    __libc_lock_unlock (lock);						      \
    return (code);							      \
  } while (0)

#define RETURN_RESTORE_HANDLER(code)					      \
  do {									      \
    /* Restore old action handler for alarm.  We don't need to know	      \
       about the current one.  */					      \
    __sigaction (SIGALRM, &saved_act, NULL);				      \
    RETURN_CLOSE_FD (code);						      \
  } while (0)

#define RETURN_CLEAR_ALARM(code)					      \
  do {									      \
    /* Clear alarm.  */							      \
    alarm (0);								      \
    /* Restore old set of handled signals.  We don't need to know	      \
       about the current one.*/						      \
    __sigprocmask (SIG_SETMASK, &saved_set, NULL);			      \
    RETURN_RESTORE_HANDLER (code);					      \
  } while (0)


int
__lckpwdf (void)
{
  sigset_t saved_set;			/* Saved set of caught signals.  */
  struct sigaction saved_act;		/* Saved signal action.  */
  sigset_t new_set;			/* New set of caught signals.  */
  struct sigaction new_act;		/* New signal action.  */
  struct flock fl;			/* Information struct for locking.  */
  int result;

  if (lock_fd != -1)
    /* Still locked by own process.  */
    return -1;

  /* Prevent problems caused by multiple threads.  */
  __libc_lock_lock (lock);

  int oflags = O_WRONLY | O_CREAT | O_CLOEXEC;
  lock_fd = __open (PWD_LOCKFILE, oflags, 0600);
  if (lock_fd == -1)
    /* Cannot create lock file.  */
    RETURN_CLOSE_FD (-1);

  /* Now we have to get exclusive write access.  Since multiple
     process could try this we won't stop when it first fails.
     Instead we set a timeout for the system call.  Once the timer
     expires it is likely that there are some problems which cannot be
     resolved by waiting.

     It is important that we don't change the signal state.  We must
     restore the old signal behaviour.  */
  memset (&new_act, '\0', sizeof (struct sigaction));
  new_act.sa_handler = noop_handler;
  __sigfillset (&new_act.sa_mask);
  new_act.sa_flags = 0ul;

  /* Install new action handler for alarm and save old.  */
  if (__sigaction (SIGALRM, &new_act, &saved_act) < 0)
    /* Cannot install signal handler.  */
    RETURN_CLOSE_FD (-1);

  /* Now make sure the alarm signal is not blocked.  */
  __sigemptyset (&new_set);
  __sigaddset (&new_set, SIGALRM);
  if (__sigprocmask (SIG_UNBLOCK, &new_set, &saved_set) < 0)
    RETURN_RESTORE_HANDLER (-1);

  /* Start timer.  If we cannot get the lock in the specified time we
     get a signal.  */
  alarm (TIMEOUT);

  /* Try to get the lock.  */
  memset (&fl, '\0', sizeof (struct flock));
  fl.l_type = F_WRLCK;
  fl.l_whence = SEEK_SET;
  result = __fcntl (lock_fd, F_SETLKW, &fl);

  RETURN_CLEAR_ALARM (result);
}
weak_alias (__lckpwdf, lckpwdf)


int
__ulckpwdf (void)
{
  int result;

  if (lock_fd == -1)
    /* There is no lock set.  */
    result = -1;
  else
    {
      /* Prevent problems caused by multiple threads.  */
      __libc_lock_lock (lock);

      result = __close (lock_fd);

      /* Mark descriptor as unused.  */
      lock_fd = -1;

      /* Clear mutex.  */
      __libc_lock_unlock (lock);
    }

  return result;
}
weak_alias (__ulckpwdf, ulckpwdf)


static void
noop_handler (int sig)
{
  /* We simply return which makes the `fcntl' call return with an error.  */
}
