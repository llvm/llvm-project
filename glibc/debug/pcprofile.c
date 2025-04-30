/* Profile PC and write result to FIFO.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1999.

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
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

/* Nonzero if we are actually doing something.  */
static int active;

/* The file descriptor of the FIFO.  */
static int fd;


static void
__attribute__ ((constructor))
install (void)
{
  /* See whether the environment variable `PCPROFILE_OUTPUT' is defined.
     If yes, it should name a FIFO.  We open it and mark ourself as active.  */
  const char *outfile = getenv ("PCPROFILE_OUTPUT");

  if (outfile != NULL && *outfile != '\0')
    {
      fd = open (outfile, O_RDWR | O_CREAT, 0666);

      if (fd != -1)
	{
	  uint32_t word;

	  active = 1;

	  /* Write a magic word which tells the reader about the byte
	     order and the size of the following entries.  */
	  word = 0xdeb00000 | sizeof (void *);
	  if (TEMP_FAILURE_RETRY (write (fd, &word, 4)) != 4)
	    {
	      /* If even this fails we shouldn't try further.  */
	      close (fd);
	      fd = -1;
	      active = 0;
	    }
	}
    }
}


static void
__attribute__ ((destructor))
uninstall (void)
{
  if (active)
    close (fd);
}


void
__cyg_profile_func_enter (void *this_fn, void *call_site)
{
  void *buf[2];

  if (! active)
    return;

  /* Now write out the current position and that of the caller.  We do
     this now, and don't cache the because we want real-time output.  */
  buf[0] = this_fn;
  buf[1] = call_site;

  write (fd, buf, sizeof buf);
}
/* We don't handle entry and exit differently here.  */
strong_alias (__cyg_profile_func_enter, __cyg_profile_func_exit)
