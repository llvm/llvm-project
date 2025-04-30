/* Test getaddrinfo and getnameinfo without usable libidn2.
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

#define TEST_USE_UTF8 1
#include "tst-resolv-ai_idn-common.c"

#include <locale.h>
#include <support/xdlfcn.h>

/* Tests for getaddrinfo.  */
static void
gai_tests (void)
{
  /* No CNAME.  */
  check_ai ("non-idn.example", 0,
            "address: STREAM/TCP 192.0.2.110 80\n");
  check_ai ("non-idn.example", AI_IDN,
            "flags: AI_IDN\n"
            "address: STREAM/TCP 192.0.2.110 80\n");
  check_ai ("non-idn.example", AI_IDN | AI_CANONNAME | AI_CANONIDN,
            "flags: AI_CANONNAME AI_IDN AI_CANONIDN\n"
            "canonname: non-idn.example\n"
            "address: STREAM/TCP 192.0.2.110 80\n");

  /* This gets passed over the network to the server, so it will
     result in an NXDOMAIN error.  */
  check_ai (NAEMCHEN ".example", 0,
            "error: Name or service not known\n");
  /* Due to missing libidn2, this fails inside getaddrinfo.  */
  check_ai (NAEMCHEN ".example", AI_IDN,
            "error: Parameter string not correctly encoded\n");
  check_ai (NAEMCHEN ".example", AI_IDN | AI_CANONNAME | AI_CANONIDN,
            "error: Parameter string not correctly encoded\n");

  /* Non-IDN CNAME.  */
  check_ai ("with.cname.example", 0,
            "address: STREAM/TCP 192.0.2.119 80\n");
  check_ai ("with.cname.example", AI_IDN,
            "flags: AI_IDN\n"
            "address: STREAM/TCP 192.0.2.119 80\n");
  check_ai ("with.cname.example", AI_IDN | AI_CANONNAME | AI_CANONIDN,
            "flags: AI_CANONNAME AI_IDN AI_CANONIDN\n"
            "canonname: non-idn-cname.example\n"
            "address: STREAM/TCP 192.0.2.119 80\n");

  /* IDN CNAME.  */
  check_ai ("With.idn-cname.example", 0,
            "address: STREAM/TCP 192.0.2.87 80\n");
  check_ai ("With.idn-cname.example", AI_IDN,
            "flags: AI_IDN\n"
            "address: STREAM/TCP 192.0.2.87 80\n");
  check_ai ("With.idn-cname.example", AI_IDN | AI_CANONNAME,
            "flags: AI_CANONNAME AI_IDN\n"
            "canonname: " ANDERES_NAEMCHEN_IDNA ".example\n"
            "address: STREAM/TCP 192.0.2.87 80\n");
  check_ai ("With.idn-cname.example",
            AI_IDN | AI_CANONNAME | AI_CANONIDN,
            "flags: AI_CANONNAME AI_IDN AI_CANONIDN\n"
            "canonname: " ANDERES_NAEMCHEN_IDNA ".example\n"
            "address: STREAM/TCP 192.0.2.87 80\n");

  /* Non-IDN to IDN CNAME chain.  */
  check_ai ("both.cname.idn-cname.example", 0,
            "address: STREAM/TCP 192.0.2.98 80\n");
  check_ai ("both.cname.idn-cname.example", AI_IDN,
            "flags: AI_IDN\n"
            "address: STREAM/TCP 192.0.2.98 80\n");
  check_ai ("both.cname.idn-cname.example", AI_IDN | AI_CANONNAME,
            "flags: AI_CANONNAME AI_IDN\n"
            "canonname: " ANDERES_NAEMCHEN_IDNA ".example\n"
            "address: STREAM/TCP 192.0.2.98 80\n");
  check_ai ("both.cname.idn-cname.example",
            AI_IDN | AI_CANONNAME | AI_CANONIDN,
            "flags: AI_CANONNAME AI_IDN AI_CANONIDN\n"
            "canonname: " ANDERES_NAEMCHEN_IDNA ".example\n"
            "address: STREAM/TCP 192.0.2.98 80\n");
}

/* Tests for getnameinfo.  */
static void
gni_tests (void)
{
  /* All non-IDN an IDN results are the same due to lack of libidn2
     support.  */
  for (int do_ni_idn = 0; do_ni_idn < 2; ++do_ni_idn)
    {
      int flags = 0;
      if (do_ni_idn)
        flags |= NI_IDN;

      gni_test (gni_non_idn_name, flags, "non-idn.example");
      gni_test (gni_non_idn_name, flags | NI_NUMERICHOST, "192.0.2.0");
      gni_test (gni_non_idn_cname_to_non_idn_name, flags,
                "non-idn-name.example");
      gni_test (gni_non_idn_cname_to_idn_name, flags,
                NAEMCHEN_IDNA ".example");
      gni_test (gni_idn_name, flags, NAEMCHEN_IDNA ".example");
      gni_test (gni_idn_cname_to_non_idn_name, flags, "non-idn-name.example");
      gni_test (gni_idn_cname_to_idn_name, flags,
                ANDERES_NAEMCHEN_IDNA ".example");

      /* Test encoding errors.  */
      gni_test (gni_invalid_idn_1, flags, "xn---.example");
      gni_test (gni_invalid_idn_2, flags, "xn--x.example");
}
}

static int
do_test (void)
{
  void *handle = xdlopen ("tst-no-libidn2.so", RTLD_LAZY);
  {
    /* Verify that this replaced libidn2.  */
    void *handle2 = xdlopen (LIBIDN2_SONAME, RTLD_LAZY | RTLD_NOLOAD);
    TEST_VERIFY (handle2 == handle);
    xdlclose (handle2);
  }

  if (setlocale (LC_CTYPE, "en_US.UTF-8") == NULL)
    FAIL_EXIT1 ("setlocale: %m");

  struct resolv_test *aux = resolv_test_start
    ((struct resolv_redirect_config)
     {
       .response_callback = response,
     });

  gai_tests ();
  gni_tests ();

  resolv_test_end (aux);
  xdlclose (handle);
  return 0;
}

#include <support/test-driver.c>
