/* Test for endgrent changing errno for BZ #24696
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

#include <stdlib.h>
#include <sys/types.h>
#include <grp.h>
#include <unistd.h>
#include <errno.h>

#include <support/check.h>
#include <support/support.h>

/* The following test verifies that if the db NSS Service is initialized
   with no database (getgrent), that a subsequent closure (endgrent) does
   not set errno. In the case of the db service it is not an error to close
   the service and so it should not set errno.  */

static int
do_test (void)
{
  /* Just make sure it's not there, although usually it won't be.  */
  unlink ("/var/db/group.db");

  /* This, in conjunction with the testroot's nsswitch.conf, causes
     the nss_db module to be "connected" and initialized - but the
     testroot has no group.db, so no mapping will be created.  */
  getgrent ();

  errno = 0;

  /* Before the fix, this would call munmap (NULL) and set errno.  */
  endgrent ();

  if (errno != 0)
    FAIL_EXIT1 ("endgrent set errno to %d\n", errno);

  return 0;
}
#include <support/test-driver.c>
