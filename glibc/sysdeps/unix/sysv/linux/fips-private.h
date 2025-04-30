/* FIPS compliance status test for GNU/Linux systems.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

#ifndef _FIPS_PRIVATE_H
#define _FIPS_PRIVATE_H

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <not-cancel.h>
#include <stdbool.h>

/* Return true if FIPS mode is enabled.  See
   sysdeps/generic/fips-private.h for more information.  */

static bool
fips_enabled_p (void)
{
  static enum
  {
    FIPS_UNTESTED = 0,
    FIPS_ENABLED = 1,
    FIPS_DISABLED = -1,
    FIPS_TEST_FAILED = -2
  } checked;

  if (checked == FIPS_UNTESTED)
    {
      int fd = __open_nocancel ("/proc/sys/crypto/fips_enabled", O_RDONLY);

      if (fd != -1)
	{
	  /* This is more than enough, the file contains a single integer.  */
	  char buf[32];
	  ssize_t n;
	  n = TEMP_FAILURE_RETRY (__read_nocancel (fd, buf, sizeof (buf) - 1));
	  __close_nocancel_nostatus (fd);

	  if (n > 0)
	    {
	      /* Terminate the string.  */
	      buf[n] = '\0';

	      char *endp;
	      long int res = strtol (buf, &endp, 10);
	      if (endp != buf && (*endp == '\0' || *endp == '\n'))
		checked = (res > 0) ? FIPS_ENABLED : FIPS_DISABLED;
	    }
	}

      if (checked == FIPS_UNTESTED)
	checked = FIPS_TEST_FAILED;
    }

  return checked == FIPS_ENABLED;
}

#endif /* _FIPS_PRIVATE_H */
