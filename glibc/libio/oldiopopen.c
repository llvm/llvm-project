/* Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Written by Per Bothner <bothner@cygnus.com>.

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
   <https://www.gnu.org/licenses/>.

   As a special exception, if you link the code in this file with
   files compiled with a GNU compiler to produce an executable,
   that does not cause the resulting executable to be covered by
   the GNU Lesser General Public License.  This exception does not
   however invalidate any other reasons why the executable file
   might be covered by the GNU Lesser General Public License.
   This exception applies to code released by its copyright holders
   in files containing the exception.  */

#define _IO_USE_OLD_IO_FILE
#include "libioP.h"
#include <signal.h>
#include <unistd.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

#include <shlib-compat.h>
#if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_1)

struct _IO_proc_file
{
  struct _IO_FILE_complete_plus file;
  /* Following fields must match those in class procbuf (procbuf.h) */
  pid_t pid;
  struct _IO_proc_file *next;
};
typedef struct _IO_proc_file _IO_proc_file;

static struct _IO_proc_file *old_proc_file_chain;

#ifdef _IO_MTSAFE_IO
static _IO_lock_t proc_file_chain_lock = _IO_lock_initializer;

static void
unlock (void *not_used)
{
  _IO_lock_unlock (proc_file_chain_lock);
}
#endif

FILE *
attribute_compat_text_section
_IO_old_proc_open (FILE *fp, const char *command, const char *mode)
{
  volatile int read_or_write;
  volatile int parent_end, child_end;
  int pipe_fds[2];
  pid_t child_pid;
  if (_IO_file_is_open (fp))
    return NULL;
  if (__pipe (pipe_fds) < 0)
    return NULL;
  if (mode[0] == 'r' && mode[1] == '\0')
    {
      parent_end = pipe_fds[0];
      child_end = pipe_fds[1];
      read_or_write = _IO_NO_WRITES;
    }
  else if (mode[0] == 'w' && mode[1] == '\0')
    {
      parent_end = pipe_fds[1];
      child_end = pipe_fds[0];
      read_or_write = _IO_NO_READS;
    }
  else
    {
      __close (pipe_fds[0]);
      __close (pipe_fds[1]);
      __set_errno (EINVAL);
      return NULL;
    }
  ((_IO_proc_file *) fp)->pid = child_pid = __fork ();
  if (child_pid == 0)
    {
      int child_std_end = mode[0] == 'r' ? 1 : 0;
      struct _IO_proc_file *p;

      __close (parent_end);
      if (child_end != child_std_end)
	{
	  __dup2 (child_end, child_std_end);
	  __close (child_end);
	}
      /* POSIX.2:  "popen() shall ensure that any streams from previous
         popen() calls that remain open in the parent process are closed
	 in the new child process." */
      for (p = old_proc_file_chain; p; p = p->next)
	__close (_IO_fileno ((FILE *) p));

      execl ("/bin/sh", "sh", "-c", command, (char *) 0);
      _exit (127);
    }
  __close (child_end);
  if (child_pid < 0)
    {
      __close (parent_end);
      return NULL;
    }
  _IO_fileno (fp) = parent_end;

  /* Link into old_proc_file_chain. */
#ifdef _IO_MTSAFE_IO
  _IO_cleanup_region_start_noarg (unlock);
  _IO_lock_lock (proc_file_chain_lock);
#endif
  ((_IO_proc_file *) fp)->next = old_proc_file_chain;
  old_proc_file_chain = (_IO_proc_file *) fp;
#ifdef _IO_MTSAFE_IO
  _IO_lock_unlock (proc_file_chain_lock);
  _IO_cleanup_region_end (0);
#endif

  _IO_mask_flags (fp, read_or_write, _IO_NO_READS|_IO_NO_WRITES);
  return fp;
}

FILE *
attribute_compat_text_section
_IO_old_popen (const char *command, const char *mode)
{
  struct locked_FILE
  {
    struct _IO_proc_file fpx;
#ifdef _IO_MTSAFE_IO
    _IO_lock_t lock;
#endif
  } *new_f;
  FILE *fp;

  new_f = (struct locked_FILE *) malloc (sizeof (struct locked_FILE));
  if (new_f == NULL)
    return NULL;
#ifdef _IO_MTSAFE_IO
  new_f->fpx.file.file._file._lock = &new_f->lock;
#endif
  fp = &new_f->fpx.file.file._file;
  _IO_old_init (fp, 0);
  _IO_JUMPS_FILE_plus (&new_f->fpx.file) = &_IO_old_proc_jumps;
  _IO_old_file_init_internal ((struct _IO_FILE_plus *) &new_f->fpx.file);
  if (_IO_old_proc_open (fp, command, mode) != NULL)
    return fp;
  _IO_un_link ((struct _IO_FILE_plus *) &new_f->fpx.file);
  free (new_f);
  return NULL;
}

int
attribute_compat_text_section
_IO_old_proc_close (FILE *fp)
{
  /* This is not name-space clean. FIXME! */
  int wstatus;
  _IO_proc_file **ptr = &old_proc_file_chain;
  pid_t wait_pid;
  int status = -1;

  /* Unlink from old_proc_file_chain. */
#ifdef _IO_MTSAFE_IO
  _IO_cleanup_region_start_noarg (unlock);
  _IO_lock_lock (proc_file_chain_lock);
#endif
  for ( ; *ptr != NULL; ptr = &(*ptr)->next)
    {
      if (*ptr == (_IO_proc_file *) fp)
	{
	  *ptr = (*ptr)->next;
	  status = 0;
	  break;
	}
    }
#ifdef _IO_MTSAFE_IO
  _IO_lock_unlock (proc_file_chain_lock);
  _IO_cleanup_region_end (0);
#endif

  if (status < 0 || __close (_IO_fileno(fp)) < 0)
    return -1;
  /* POSIX.2 Rationale:  "Some historical implementations either block
     or ignore the signals SIGINT, SIGQUIT, and SIGHUP while waiting
     for the child process to terminate.  Since this behavior is not
     described in POSIX.2, such implementations are not conforming." */
  do
    {
      wait_pid = __waitpid (((_IO_proc_file *) fp)->pid, &wstatus, 0);
    }
  while (wait_pid == -1 && errno == EINTR);
  if (wait_pid == -1)
    return -1;
  return wstatus;
}

const struct _IO_jump_t _IO_old_proc_jumps libio_vtable = {
  JUMP_INIT_DUMMY,
  JUMP_INIT(finish, _IO_old_file_finish),
  JUMP_INIT(overflow, _IO_old_file_overflow),
  JUMP_INIT(underflow, _IO_old_file_underflow),
  JUMP_INIT(uflow, _IO_default_uflow),
  JUMP_INIT(pbackfail, _IO_default_pbackfail),
  JUMP_INIT(xsputn, _IO_old_file_xsputn),
  JUMP_INIT(xsgetn, _IO_default_xsgetn),
  JUMP_INIT(seekoff, _IO_old_file_seekoff),
  JUMP_INIT(seekpos, _IO_default_seekpos),
  JUMP_INIT(setbuf, _IO_old_file_setbuf),
  JUMP_INIT(sync, _IO_old_file_sync),
  JUMP_INIT(doallocate, _IO_file_doallocate),
  JUMP_INIT(read, _IO_file_read),
  JUMP_INIT(write, _IO_old_file_write),
  JUMP_INIT(seek, _IO_file_seek),
  JUMP_INIT(close, _IO_old_proc_close),
  JUMP_INIT(stat, _IO_file_stat),
  JUMP_INIT(showmanyc, _IO_default_showmanyc),
  JUMP_INIT(imbue, _IO_default_imbue)
};

strong_alias (_IO_old_popen, __old_popen)
compat_symbol (libc, _IO_old_popen, _IO_popen, GLIBC_2_0);
compat_symbol (libc, __old_popen, popen, GLIBC_2_0);
compat_symbol (libc, _IO_old_proc_open, _IO_proc_open, GLIBC_2_0);
compat_symbol (libc, _IO_old_proc_close, _IO_proc_close, GLIBC_2_0);

#endif
