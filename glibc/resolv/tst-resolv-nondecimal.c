/* Test name resolution behavior for octal, hexadecimal IPv4 addresses.
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

#include <netdb.h>
#include <stdlib.h>
#include <support/check.h>
#include <support/check_nss.h>
#include <support/resolv_test.h>
#include <support/support.h>

static void
response (const struct resolv_response_context *ctx,
          struct resolv_response_builder *b,
          const char *qname, uint16_t qclass, uint16_t qtype)
{
  /* The tests are not supposed send any DNS queries.  */
  FAIL_EXIT1 ("unexpected DNS query for %s/%d/%d", qname, qclass, qtype);
}

static void
run_query_addrinfo (const char *query, const char *address)
{
  char *quoted_query = support_quote_string (query);

  struct addrinfo *ai;
  struct addrinfo hints =
    {
     .ai_socktype = SOCK_STREAM,
     .ai_protocol = IPPROTO_TCP,
    };

  char *context = xasprintf ("getaddrinfo \"%s\" AF_INET", quoted_query);
  char *expected = xasprintf ("address: STREAM/TCP %s 80\n", address);
  hints.ai_family = AF_INET;
  int ret = getaddrinfo (query, "80", &hints, &ai);
  check_addrinfo (context, ai, ret, expected);
  if (ret == 0)
    freeaddrinfo (ai);
  free (context);

  context = xasprintf ("getaddrinfo \"%s\" AF_UNSPEC", quoted_query);
  hints.ai_family = AF_UNSPEC;
  ret = getaddrinfo (query, "80", &hints, &ai);
  check_addrinfo (context, ai, ret, expected);
  if (ret == 0)
    freeaddrinfo (ai);
  free (expected);
  free (context);

  context = xasprintf ("getaddrinfo \"%s\" AF_INET6", quoted_query);
  expected = xasprintf ("flags: AI_V4MAPPED\n"
                        "address: STREAM/TCP ::ffff:%s 80\n",
                        address);
  hints.ai_family = AF_INET6;
  hints.ai_flags = AI_V4MAPPED;
  ret = getaddrinfo (query, "80", &hints, &ai);
  check_addrinfo (context, ai, ret, expected);
  if (ret == 0)
    freeaddrinfo (ai);
  free (expected);
  free (context);

  free (quoted_query);
}

static void
run_query (const char *query, const char *address)
{
  char *quoted_query = support_quote_string (query);
  char *context = xasprintf ("gethostbyname (\"%s\")", quoted_query);
  char *expected = xasprintf ("name: %s\n"
                              "address: %s\n", query, address);
  check_hostent (context, gethostbyname (query), expected);
  free (context);

  context = xasprintf ("gethostbyname_r \"%s\"", quoted_query);
  struct hostent storage;
  char buf[4096];
  struct hostent *e = NULL;
  TEST_COMPARE (gethostbyname_r (query, &storage, buf, sizeof (buf),
                                 &e, &h_errno), 0);
  check_hostent (context, e, expected);
  free (context);

  context = xasprintf ("gethostbyname2 (\"%s\", AF_INET)", quoted_query);
  check_hostent (context, gethostbyname2 (query, AF_INET), expected);
  free (context);

  context = xasprintf ("gethostbyname2_r \"%s\" AF_INET", quoted_query);
  e = NULL;
  TEST_COMPARE (gethostbyname2_r (query, AF_INET, &storage, buf, sizeof (buf),
                                  &e, &h_errno), 0);
  check_hostent (context, e, expected);
  free (context);
  free (expected);

  free (quoted_query);

  /* The gethostbyname tests are always valid for getaddrinfo, but not
     vice versa.  */
  run_query_addrinfo (query, address);
}

static int
do_test (void)
{
  struct resolv_test *aux = resolv_test_start
    ((struct resolv_redirect_config)
     {
       .response_callback = response,
     });

  run_query ("192.000.002.010", "192.0.2.8");

  /* Hexadecimal numbers are not accepted by gethostbyname.  */
  run_query_addrinfo ("0xc0000210", "192.0.2.16");
  run_query_addrinfo ("192.0x234", "192.0.2.52");

  resolv_test_end (aux);

  return 0;
}

#include <support/test-driver.c>
