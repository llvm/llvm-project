/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
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

#include <fcntl.h>
#include <paths.h>
#include <unistd.h>

#include <hurd.h>
#include <hurd/fd.h>

#include <set-hooks.h>

/* Try to get a machine dependent instruction which will make the
   program crash.  This is used in case everything else fails.  */
#include <abort-instr.h>
#ifndef ABORT_INSTRUCTION
/* No such instruction is available.  */
# define ABORT_INSTRUCTION
#endif

static void
check_one_fd (int fd, int mode)
{
  struct hurd_fd *d;

  d = _hurd_fd_get (fd);
  if (d == NULL)
    {
      /* This descriptor hasn't been opened.  We try to allocate the
         descriptor and open /dev/null on it so that the SUID program
         we are about to start does not accidentally use this
         descriptor.  */
      d = _hurd_alloc_fd (NULL, fd);
      if (d != NULL)
	{
	  mach_port_t port;

	  port = __file_name_lookup (_PATH_DEVNULL, mode, 0);
	  if (port)
	    {
	      /* Since /dev/null isn't supposed to be a terminal, we
		 avoid any ctty magic.  */
	      d->port.port = port;
	      d->flags = 0;

	      __spin_unlock (&d->port.lock);
	      return;
	    }
	}

      /* We cannot even give an error message here since it would run
	 into the same problems.  */
      while (1)
	/* Try for ever and ever.  */
	ABORT_INSTRUCTION;
    }
}

static void
check_standard_fds (void)
{
  /* Check all three standard file descriptors.  */
  check_one_fd (STDIN_FILENO, O_RDONLY);
  check_one_fd (STDOUT_FILENO, O_RDWR);
  check_one_fd (STDERR_FILENO, O_RDWR);
}

static void
init_standard_fds (void)
{
  /* Now that we have FDs, make sure that, if this is a SUID program,
     FDs 0, 1 and 2 are allocated.  If necessary we'll set them up
     ourselves.  If that's not possible we stop the program.  */
  if (__builtin_expect (__libc_enable_secure, 0))
    check_standard_fds ();

  (void) &init_standard_fds;	/* Avoid "defined but not used" warning.  */
}
text_set_element (_hurd_fd_subinit, init_standard_fds);


#ifndef SHARED
void
__libc_check_standard_fds (void)
{
  /* We don't check the standard file descriptors here.  They will be
     checked when we initialize the file descriptor table, as part of
     the _hurd_fd_subinit hook.

     This function is only present to make sure that this module gets
     linked in when part of the static libc.  */
}
#endif
