/* Tests for the support_format_dns_packet function.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include <support/check.h>
#include <support/format_nss.h>
#include <support/run_diff.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void
check_packet (const void *buffer, size_t length,
              const char *name, const char *expected)
{
  char *actual = support_format_dns_packet (buffer, length);
  if (strcmp (actual, expected) != 0)
    {
      support_record_failure ();
      printf ("error: formatted packet does not match: %s\n", name);
      support_run_diff ("expected", expected,
                        "actual", actual);
    }
  free (actual);
}

static void
test_aaaa_length (void)
{
  static const char packet[] =
    /* Header: Response with two records.  */
    "\x12\x34\x80\x00\x00\x01\x00\x02\x00\x00\x00\x00"
    /* Question section.  www.example/IN/AAAA.  */
    "\x03www\x07""example\x00\x00\x1c\x00\x01"
    /* Answer section.  www.example AAAA [corrupted].  */
    "\xc0\x0c"
    "\x00\x1c\x00\x01\x00\x00\x00\x00\x00\x10"
    "\x20\x01\x0d\xb8\x05\x06\x07\x08"
    "\x11\x12\x13\x14\x15\x16\x17\x18"
    /* www.example AAAA [corrupted].  */
    "\xc0\x0c"
    "\x00\x1c\x00\x01\x00\x00\x00\x00\x00\x11"
    "\x01\x02\x03\x04\x05\x06\x07\x08"
    "\x11\x12\x13\x14\x15\x16\x17\x18" "\xff";
  check_packet (packet, sizeof (packet) - 1, __func__,
                "name: www.example\n"
                "address: 2001:db8:506:708:1112:1314:1516:1718\n"
                "error: AAAA record of size 17: www.example\n");
}

static void
test_multiple_cnames (void)
{
  static const char packet[] =
    /* Header: Response with three records.  */
    "\x12\x34\x80\x00\x00\x01\x00\x03\x00\x00\x00\x00"
    /* Question section.  www.example/IN/A.  */
    "\x03www\x07""example\x00\x00\x01\x00\x01"
    /* Answer section.  www.example CNAME www1.example.  */
    "\xc0\x0c"
    "\x00\x05\x00\x01\x00\x00\x00\x00\x00\x07"
    "\x04www1\xc0\x10"
    /* www1 CNAME www2.  */
    "\x04www1\xc0\x10"
    "\x00\x05\x00\x01\x00\x00\x00\x00\x00\x07"
    "\x04www2\xc0\x10"
    /* www2 A 192.0.2.1.  */
    "\x04www2\xc0\x10"
    "\x00\x01\x00\x01\x00\x00\x00\x00\x00\x04"
    "\xc0\x00\x02\x01";
  check_packet (packet, sizeof (packet) - 1, __func__,
                "name: www.example\n"
                "name: www1.example\n"
                "name: www2.example\n"
                "address: 192.0.2.1\n");
}

static int
do_test (void)
{
  test_aaaa_length ();
  test_multiple_cnames ();
  return 0;
}

#include <support/test-driver.c>
