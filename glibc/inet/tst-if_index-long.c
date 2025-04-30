/* Check for descriptor leak in if_nametoindex with a long interface name.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

/* This test checks for a descriptor leak in case of a long interface
   name (CVE-2018-19591, bug 23927).  */

#include <errno.h>
#include <net/if.h>
#include <netdb.h>
#include <string.h>
#include <support/check.h>
#include <support/descriptors.h>
#include <support/support.h>

static int
do_test (void)
{
  struct support_descriptors *descrs = support_descriptors_list ();

  /* Prepare a name which is just as long as required for trigging the
     bug.  */
  char name[IFNAMSIZ + 1];
  memset (name, 'A', IFNAMSIZ);
  name[IFNAMSIZ] = '\0';
  TEST_COMPARE (strlen (name), IFNAMSIZ);
  struct ifreq ifr;
  TEST_COMPARE (strlen (name), sizeof (ifr.ifr_name));

  /* Test directly via if_nametoindex.  */
  TEST_COMPARE (if_nametoindex (name), 0);
  TEST_COMPARE (errno, ENODEV);
  support_descriptors_check (descrs);

  /* Same test via getaddrinfo.  */
  char *host = xasprintf ("fea0::%%%s", name);
  struct addrinfo hints = { .ai_flags = AI_NUMERICHOST, };
  struct addrinfo *ai;
  TEST_COMPARE (getaddrinfo (host, NULL, &hints, &ai), EAI_NONAME);
  support_descriptors_check (descrs);

  support_descriptors_free (descrs);

  return 0;
}

#include <support/test-driver.c>
