/* Test the behavior of the trust-ad option.
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

#include <resolv.h>
#include <stdlib.h>
#include <string.h>
#include <support/check.h>
#include <support/check_nss.h>
#include <support/resolv_test.h>
#include <support/support.h>

/* This controls properties of the response.  volatile because
   __res_send is incorrectly declared as __THROW.  */
static volatile unsigned char response_number;
static volatile bool response_ad_bit;
static volatile bool query_ad_bit;

static void
response (const struct resolv_response_context *ctx,
          struct resolv_response_builder *b,
          const char *qname, uint16_t qclass, uint16_t qtype)
{
  TEST_COMPARE (qclass, C_IN);
  TEST_COMPARE (qtype, T_A);
  TEST_COMPARE_STRING (qname, "www.example");

  HEADER header;
  memcpy (&header, ctx->query_buffer, sizeof (header));
  TEST_COMPARE (header.ad, query_ad_bit);

  struct resolv_response_flags flags = { .ad = response_ad_bit, };
  resolv_response_init (b, flags);
  resolv_response_add_question (b, qname, qclass, qtype);
  resolv_response_section (b, ns_s_an);
  resolv_response_open_record (b, qname, qclass, T_A, 0x12345678);
  char addr[4] = { 192, 0, 2, response_number };
  resolv_response_add_data (b, addr, sizeof (addr));
  resolv_response_close_record (b);
}

static void
check_answer (const unsigned char *buffer, size_t buffer_length,
              bool expected_ad)
{
  HEADER header;
  TEST_VERIFY (buffer_length > sizeof (header));
  memcpy (&header, buffer, sizeof (header));
  TEST_COMPARE (0, header.aa);
  TEST_COMPARE (expected_ad, header.ad);
  TEST_COMPARE (0, header.opcode);
  TEST_COMPARE (1, header.qr);
  TEST_COMPARE (0, header.rcode);
  TEST_COMPARE (1, header.rd);
  TEST_COMPARE (0, header.tc);
  TEST_COMPARE (1, ntohs (header.qdcount));
  TEST_COMPARE (1, ntohs (header.ancount));
  TEST_COMPARE (0, ntohs (header.nscount));
  TEST_COMPARE (0, ntohs (header.arcount));

  char *description = xasprintf ("response=%d ad=%d",
                                 response_number, expected_ad);
  char *expected = xasprintf ("name: www.example\n"
                              "address: 192.0.2.%d\n", response_number);
  check_dns_packet (description, buffer, buffer_length, expected);
  free (expected);
  free (description);
}

static int
do_test (void)
{
  struct resolv_test *aux = resolv_test_start
    ((struct resolv_redirect_config)
     {
       .response_callback = response,
     });

  /* By default, the resolver is not trusted, and the AD bit is
     cleared.  */

  static const unsigned char hand_crafted_query[] =
    {
     10, 11,                    /* Transaction ID.  */
     1, 0x20,                   /* Query with RD, AD flags.  */
     0, 1,                      /* One question.  */
     0, 0, 0, 0, 0, 0,          /* The other sections are empty.  */
     3, 'w', 'w', 'w', 7, 'e', 'x', 'a', 'm', 'p', 'l', 'e', 0,
     0, T_A,                    /* A query.  */
     0, 1,                      /* Class IN.  */
    };

  ++response_number;
  response_ad_bit = false;

  unsigned char buffer[512];
  memset (buffer, 255, sizeof (buffer));
  query_ad_bit = true;
  int ret = res_send (hand_crafted_query, sizeof (hand_crafted_query),
                      buffer, sizeof (buffer));
  TEST_VERIFY (ret > 0);
  check_answer (buffer, ret, false);

  ++response_number;
  memset (buffer, 255, sizeof (buffer));
  query_ad_bit = false;
  ret = res_query ("www.example", C_IN, T_A, buffer, sizeof (buffer));
  TEST_VERIFY (ret > 0);
  check_answer (buffer, ret, false);
  response_ad_bit = true;

  response_ad_bit = true;

  ++response_number;
  query_ad_bit = true;
  ret = res_send (hand_crafted_query, sizeof (hand_crafted_query),
                  buffer, sizeof (buffer));
  TEST_VERIFY (ret > 0);
  check_answer (buffer, ret, false);

  ++response_number;
  memset (buffer, 255, sizeof (buffer));
  query_ad_bit = false;
  ret = res_query ("www.example", C_IN, T_A, buffer, sizeof (buffer));
  TEST_VERIFY (ret > 0);
  check_answer (buffer, ret, false);

  /* No AD bit set in generated queries.  */
  memset (buffer, 255, sizeof (buffer));
  ret = res_mkquery (QUERY, "www.example", C_IN, T_A,
                     (const unsigned char *) "", 0, NULL,
                     buffer, sizeof (buffer));
  HEADER header;
  memcpy (&header, buffer, sizeof (header));
  TEST_VERIFY (!header.ad);

  /* With RES_TRUSTAD, the AD bit is passed through if it set in the
     response.  It is also included in queries.  */

  _res.options |= RES_TRUSTAD;
  query_ad_bit = true;

  response_ad_bit = false;

  ++response_number;
  memset (buffer, 255, sizeof (buffer));
  ret = res_send (hand_crafted_query, sizeof (hand_crafted_query),
                  buffer, sizeof (buffer));
  TEST_VERIFY (ret > 0);
  check_answer (buffer, ret, false);

  ++response_number;
  memset (buffer, 255, sizeof (buffer));
  ret = res_query ("www.example", C_IN, T_A, buffer, sizeof (buffer));
  TEST_VERIFY (ret > 0);
  check_answer (buffer, ret, false);

  response_ad_bit = true;

  ++response_number;
  memset (buffer, 0, sizeof (buffer));
  ret = res_send (hand_crafted_query, sizeof (hand_crafted_query),
                  buffer, sizeof (buffer));
  TEST_VERIFY (ret > 0);
  check_answer (buffer, ret, true);

  ++response_number;
  memset (buffer, 0, sizeof (buffer));
  ret = res_query ("www.example", C_IN, T_A, buffer, sizeof (buffer));
  TEST_VERIFY (ret > 0);
  check_answer (buffer, ret, true);

  /* AD bit set in generated queries.  */
  memset (buffer, 0, sizeof (buffer));
  ret = res_mkquery (QUERY, "www.example", C_IN, T_A,
                     (const unsigned char *) "", 0, NULL,
                     buffer, sizeof (buffer));
  memcpy (&header, buffer, sizeof (header));
  TEST_VERIFY (header.ad);

  resolv_test_end (aux);

  return 0;
}

#include <support/test-driver.c>
