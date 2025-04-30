/* Test inet_pton functions.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#include <arpa/inet.h>
#include <resolv/resolv-internal.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <support/check.h>
#include <support/next_to_fault.h>
#include <support/xunistd.h>
#include <unistd.h>

struct test_case
{
  /* The input data.  */
  const char *input;

  /* True if AF_INET parses successfully.  */
  bool ipv4_ok;

  /* True if AF_INET6 parses successfully.  */
  bool ipv6_ok;

  /* Expected result for AF_INET.  */
  unsigned char ipv4_expected[4];

  /* Expected result for AF_INET6.  */
  unsigned char ipv6_expected[16];
};

static void
check_result (const char *what, const struct test_case *t, int family,
              void *result_buffer, int inet_ret)
{
  TEST_VERIFY_EXIT (inet_ret >= -1);
  TEST_VERIFY_EXIT (inet_ret <= 1);

  int ok;
  const unsigned char *expected;
  size_t result_size;
  switch (family)
    {
    case AF_INET:
      ok = t->ipv4_ok;
      expected = t->ipv4_expected;
      result_size = 4;
      break;
    case AF_INET6:
      ok = t->ipv6_ok;
      expected = t->ipv6_expected;
      result_size = 16;
      break;
    default:
      FAIL_EXIT1 ("invalid address family %d", family);
    }

  if (inet_ret != ok)
    {
      support_record_failure ();
      printf ("error: %s return value mismatch for [[%s]], family %d\n"
              "  expected: %d\n"
              "  actual: %d\n",
              what, t->input, family, ok, inet_ret);
      return;
    }
  if (memcmp (result_buffer, expected, result_size) != 0)
    {
      support_record_failure ();
      printf ("error: %s result mismatch for [[%s]], family %d\n",
              what, t->input, family);
    }
}

static void
run_one_test (const struct test_case *t)
{
  size_t test_len = strlen (t->input);

  struct support_next_to_fault ntf_out4 = support_next_to_fault_allocate (4);
  struct support_next_to_fault ntf_out6 = support_next_to_fault_allocate (16);

  /* inet_pton requires NUL termination.  */
  {
    struct support_next_to_fault ntf_in
      = support_next_to_fault_allocate (test_len + 1);
    memcpy (ntf_in.buffer, t->input, test_len + 1);
    memset (ntf_out4.buffer, 0, 4);
    check_result ("inet_pton", t, AF_INET, ntf_out4.buffer,
                  inet_pton (AF_INET, ntf_in.buffer, ntf_out4.buffer));
    memset (ntf_out6.buffer, 0, 16);
    check_result ("inet_pton", t, AF_INET6, ntf_out6.buffer,
                  inet_pton (AF_INET6, ntf_in.buffer, ntf_out6.buffer));
    support_next_to_fault_free (&ntf_in);
  }

  /* __inet_pton_length does not require NUL termination.  */
  {
    struct support_next_to_fault ntf_in
      = support_next_to_fault_allocate (test_len);
    memcpy (ntf_in.buffer, t->input, test_len);
    memset (ntf_out4.buffer, 0, 4);
    check_result ("__inet_pton_length", t, AF_INET, ntf_out4.buffer,
                  __inet_pton_length (AF_INET, ntf_in.buffer, ntf_in.length,
                                      ntf_out4.buffer));
    memset (ntf_out6.buffer, 0, 16);
    check_result ("__inet_pton_length", t, AF_INET6, ntf_out6.buffer,
                  __inet_pton_length (AF_INET6, ntf_in.buffer, ntf_in.length,
                                      ntf_out6.buffer));
    support_next_to_fault_free (&ntf_in);
  }

  support_next_to_fault_free (&ntf_out4);
  support_next_to_fault_free (&ntf_out6);
}

/* The test cases were manually crafted and the set enhanced with
   American Fuzzy Lop.  */
const struct test_case test_cases[] =
  {
    {.input = ".:", },
    {.input = "0.0.0.0",
     .ipv4_ok = true,
     .ipv4_expected = {0, 0, 0, 0},
    },
    {.input = "0.:", },
    {.input = "00", },
    {.input = "0000000", },
    {.input = "00000000000000000", },
    {.input = "092.", },
    {.input = "10.0.301.2", },
    {.input = "127.0.0.1",
     .ipv4_ok = true,
     .ipv4_expected = {127, 0, 0, 1},
    },
    {.input = "19..", },
    {.input = "192.0.2.-1", },
    {.input = "192.0.2.01", },
    {.input = "192.0.2.1.", },
    {.input = "192.0.2.1192.", },
    {.input = "192.0.2.192.\377..", },
    {.input = "192.0.2.256", },
    {.input = "192.0.2.27",
     .ipv4_ok = true,
     .ipv4_expected = {192, 0, 2, 27},
    },
    {.input = "192.0.201.", },
    {.input = "192.0.261.", },
    {.input = "192.0.2\256", },
    {.input = "192.0.\262.", },
    {.input = "192.062.", },
    {.input = "192.092.\256", },
    {.input = "192.0\2562.", },
    {.input = "192.192.0.2661\031", },
    {.input = "192.192.00n2.1.", },
    {.input = "192.192.2.190.", },
    {.input = "192.255.255.2555", },
    {.input = "192.92.219\023.", },
    {.input = "192.\260.2.", },
    {.input = "1:1::1:1",
     .ipv6_ok = true,
     .ipv6_expected = {
       0x0, 0x1, 0x0, 0x1, 0x0, 0x0, 0x0, 0x0,
       0x0, 0x0, 0x0, 0x0, 0x0, 0x1, 0x0, 0x1
     },
    },
    {.input = "2", },
    {.input = "2.", },
    {.input = "2001:db8:00001::f", },
    {.input = "2001:db8:10000::f", },
    {.input = "2001:db8:1234:5678:abcd:ef01:2345:67",
     .ipv6_ok = true,
     .ipv6_expected = {
       0x20, 0x1, 0xd, 0xb8, 0x12, 0x34, 0x56, 0x78,
       0xab, 0xcd, 0xef, 0x1, 0x23, 0x45, 0x0, 0x67
     },
    },
    {.input = "2001:db8:1234:5678:abcd:ef01:2345:6789:1", },
    {.input = "2001:db8:1234:5678:abcd:ef01:2345::6789", },
    {.input = "2001:db8::0",
     .ipv6_ok = true,
     .ipv6_expected = {
       0x20, 0x1, 0xd, 0xb8, 0x0, 0x0, 0x0, 0x0,
       0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0
     },
    },
    {.input = "2001:db8::00",
     .ipv6_ok = true,
     .ipv6_expected = {
       0x20, 0x1, 0xd, 0xb8, 0x0, 0x0, 0x0, 0x0,
       0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0
     },
    },
    {.input = "2001:db8::1",
     .ipv6_ok = true,
     .ipv6_expected = {
       0x20, 0x1, 0xd, 0xb8, 0x0, 0x0, 0x0, 0x0,
       0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1
     },
    },
    {.input = "2001:db8::10",
     .ipv6_ok = true,
     .ipv6_expected = {
       0x20, 0x1, 0xd, 0xb8, 0x0, 0x0, 0x0, 0x0,
       0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x10
     },
    },
    {.input = "2001:db8::19",
     .ipv6_ok = true,
     .ipv6_expected = {
       0x20, 0x1, 0xd, 0xb8, 0x0, 0x0, 0x0, 0x0,
       0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x19
     },
    },
    {.input = "2001:db8::1::\012", },
    {.input = "2001:db8::1::2\012", },
    {.input = "2001:db8::2",
     .ipv6_ok = true,
     .ipv6_expected = {
       0x20, 0x1, 0xd, 0xb8, 0x0, 0x0, 0x0, 0x0,
       0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x2
     },
    },
    {.input = "2001:db8::3",
     .ipv6_ok = true,
     .ipv6_expected = {
       0x20, 0x1, 0xd, 0xb8, 0x0, 0x0, 0x0, 0x0,
       0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x3
     },
    },
    {.input = "2001:db8::4",
     .ipv6_ok = true,
     .ipv6_expected = {
       0x20, 0x1, 0xd, 0xb8, 0x0, 0x0, 0x0, 0x0,
       0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x4
     },
    },
    {.input = "2001:db8::5",
     .ipv6_ok = true,
     .ipv6_expected = {
       0x20, 0x1, 0xd, 0xb8, 0x0, 0x0, 0x0, 0x0,
       0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x5
     },
    },
    {.input = "2001:db8::6",
     .ipv6_ok = true,
     .ipv6_expected = {
       0x20, 0x1, 0xd, 0xb8, 0x0, 0x0, 0x0, 0x0,
       0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x6
     },
    },
    {.input = "2001:db8::7",
     .ipv6_ok = true,
     .ipv6_expected = {
       0x20, 0x1, 0xd, 0xb8, 0x0, 0x0, 0x0, 0x0,
       0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x7
     },
    },
    {.input = "2001:db8::8",
     .ipv6_ok = true,
     .ipv6_expected = {
       0x20, 0x1, 0xd, 0xb8, 0x0, 0x0, 0x0, 0x0,
       0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x8
     },
    },
    {.input = "2001:db8::9",
     .ipv6_ok = true,
     .ipv6_expected = {
       0x20, 0x1, 0xd, 0xb8, 0x0, 0x0, 0x0, 0x0,
       0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x9
     },
    },
    {.input = "2001:db8::A",
     .ipv6_ok = true,
     .ipv6_expected = {
       0x20, 0x1, 0xd, 0xb8, 0x0, 0x0, 0x0, 0x0,
       0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xa
     },
    },
    {.input = "2001:db8::B",
     .ipv6_ok = true,
     .ipv6_expected = {
       0x20, 0x1, 0xd, 0xb8, 0x0, 0x0, 0x0, 0x0,
       0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xb
     },
    },
    {.input = "2001:db8::C",
     .ipv6_ok = true,
     .ipv6_expected = {
       0x20, 0x1, 0xd, 0xb8, 0x0, 0x0, 0x0, 0x0,
       0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xc
     },
    },
    {.input = "2001:db8::D",
     .ipv6_ok = true,
     .ipv6_expected = {
       0x20, 0x1, 0xd, 0xb8, 0x0, 0x0, 0x0, 0x0,
       0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xd
     },
    },
    {.input = "2001:db8::E",
     .ipv6_ok = true,
     .ipv6_expected = {
       0x20, 0x1, 0xd, 0xb8, 0x0, 0x0, 0x0, 0x0,
       0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xe
     },
    },
    {.input = "2001:db8::F",
     .ipv6_ok = true,
     .ipv6_expected = {
       0x20, 0x1, 0xd, 0xb8, 0x0, 0x0, 0x0, 0x0,
       0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf
     },
    },
    {.input = "2001:db8::a",
     .ipv6_ok = true,
     .ipv6_expected = {
       0x20, 0x1, 0xd, 0xb8, 0x0, 0x0, 0x0, 0x0,
       0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xa
     },
    },
    {.input = "2001:db8::b",
     .ipv6_ok = true,
     .ipv6_expected = {
       0x20, 0x1, 0xd, 0xb8, 0x0, 0x0, 0x0, 0x0,
       0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xb
     },
    },
    {.input = "2001:db8::c",
     .ipv6_ok = true,
     .ipv6_expected = {
       0x20, 0x1, 0xd, 0xb8, 0x0, 0x0, 0x0, 0x0,
       0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xc
     },
    },
    {.input = "2001:db8::d",
     .ipv6_ok = true,
     .ipv6_expected = {
       0x20, 0x1, 0xd, 0xb8, 0x0, 0x0, 0x0, 0x0,
       0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xd
     },
    },
    {.input = "2001:db8::e",
     .ipv6_ok = true,
     .ipv6_expected = {
       0x20, 0x1, 0xd, 0xb8, 0x0, 0x0, 0x0, 0x0,
       0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xe
     },
    },
    {.input = "2001:db8::f",
     .ipv6_ok = true,
     .ipv6_expected = {
       0x20, 0x1, 0xd, 0xb8, 0x0, 0x0, 0x0, 0x0,
       0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf
     },
    },
    {.input = "2001:db8::ff",
     .ipv6_ok = true,
     .ipv6_expected = {
       0x20, 0x1, 0xd, 0xb8, 0x0, 0x0, 0x0, 0x0,
       0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xff
     },
    },
    {.input = "2001:db8::ffff:2\012", },
    {.input = "22", },
    {.input = "2222@", },
    {.input = "255.255.255.255",
     .ipv4_ok = true,
     .ipv4_expected = {255, 255, 255, 255},
    },
    {.input = "255.255.255.255\001", },
    {.input = "255.255.255.25555", },
    {.input = "2:", },
    {.input = "2:a:8:EEEE::EEEE:F:EEE8:EEEE\034*:", },
    {.input = "2:ff:1:1:7:ff:1:1:7.", },
    {.input = "2f:0000000000000000000000000000000000000000000000000000000000"
     "0000000000000000000000000000000000000000000000000000000000000000000000"
     "0G01",
    },
    {.input = "429495", },
    {.input = "5::5::", },
    {.input = "6.6.", },
    {.input = "992.", },
    {.input = "::",
     .ipv6_ok = true,
     .ipv6_expected = {
       0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
       0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0
     },
    },
    {.input = "::00001", },
    {.input = "::1",
     .ipv6_ok = true,
     .ipv6_expected = {
       0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
       0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1
     },
    },
    {.input = "::10000", },
    {.input = "::1:1",
     .ipv6_ok = true,
     .ipv6_expected = {
       0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
       0x0, 0x0, 0x0, 0x0, 0x0, 0x1, 0x0, 0x1
     },
    },
    {.input = "::ff:1:1:7.0.0.1",
     .ipv6_ok = true,
     .ipv6_expected = {
       0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xff,
       0x0, 0x1, 0x0, 0x1, 0x7, 0x0, 0x0, 0x1
     },
    },
    {.input = "::ff:1:1:7:ff:1:1:7.", },
    {.input = "::ff:1:1:7ff:1:8:7.0.0.1", },
    {.input = "::ff:1:1:7ff:1:8f:1:1:71", },
    {.input = "::ffff:02fff:127.0.S1", },
    {.input = "::ffff:127.0.0.1",
     .ipv6_ok = true,
     .ipv6_expected = {
       0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
       0x0, 0x0, 0xff, 0xff, 0x7f, 0x0, 0x0, 0x1
     },
    },
    {.input = "::ffff:1:7.0.0.1",
     .ipv6_ok = true,
     .ipv6_expected = {
       0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
       0xff, 0xff, 0x0, 0x1, 0x7, 0x0, 0x0, 0x1
     },
    },
    {.input = ":\272", },
    {.input = "A:f:ff:1:1:D:ff:1:1::7.", },
    {.input = "AAAAA.", },
    {.input = "D:::", },
    {.input = "DF8F", },
    {.input = "F::",
     .ipv6_ok = true,
     .ipv6_expected = {
       0x0, 0xf, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
       0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0
     },
    },
    {.input = "F:A:8:EEEE:8:EEEE\034*:", },
    {.input = "F:a:8:EEEE:8:EEEE\034*:", },
    {.input = "F:ff:100:7ff:1:8:7.0.10.1",
     .ipv6_ok = true,
     .ipv6_expected = {
       0x0, 0xf, 0x0, 0xff, 0x1, 0x0, 0x7, 0xff,
       0x0, 0x1, 0x0, 0x8, 0x7, 0x0, 0xa, 0x1
     },
    },
    {.input = "d92.", },
    {.input = "ff:00000000000000000000000000000000000000000000000000000000000"
     "00000000000000000000000000000000000000000000000000000000000000000001",
    },
    {.input = "fff2:2::ff2:2:f7",
     .ipv6_ok = true,
     .ipv6_expected = {
       0xff, 0xf2, 0x0, 0x2, 0x0, 0x0, 0x0, 0x0,
       0x0, 0x0, 0xf, 0xf2, 0x0, 0x2, 0x0, 0xf7
     },
    },
    {.input = "ffff:ff:ff:fff:ff:ff:ff:", },
    {.input = "\272:", },
    {NULL}
  };

static int
do_test (void)
{
  for (size_t i = 0; test_cases[i].input != NULL; ++i)
    run_one_test (test_cases + i);
  return 0;
}

#include <support/test-driver.c>
