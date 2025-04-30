/* Test handling of binary domain names with res_send.
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

#include <resolv.h>
#include <string.h>
#include <support/check.h>
#include <support/resolv_test.h>

static void
response (const struct resolv_response_context *ctx,
          struct resolv_response_builder *b,
          const char *qname, uint16_t qclass, uint16_t qtype)
{
  TEST_COMPARE (qclass, C_IN);
  TEST_COMPARE (qtype, T_TXT);
  TEST_VERIFY (strlen (qname) <= 255);

  struct resolv_response_flags flags = { 0 };
  resolv_response_init (b, flags);
  resolv_response_add_question (b, qname, qclass, qtype);
  resolv_response_section (b, ns_s_an);
  resolv_response_open_record (b, qname, qclass, T_TXT, 0x12345678);
  unsigned char qnamelen = strlen (qname);
  resolv_response_add_data (b, &qnamelen, 1);
  resolv_response_add_data (b, qname, qnamelen);
  resolv_response_close_record (b);
}

static int
do_test (void)
{
  struct resolv_test *aux = resolv_test_start
    ((struct resolv_redirect_config)
     {
       .response_callback = response,
     });

  for (int b = 0; b <= 255; ++b)
    {
      unsigned char query[] =
        {
          b, b,                 /* Transaction ID.  */
          1, 0,                 /* Query with RD flag.  */
          0, 1,                 /* One question.  */
          0, 0, 0, 0, 0, 0,     /* The other sections are empty.  */
          1, b, 7, 'e', 'x', 'a', 'm', 'p', 'l', 'e', 0,
          0, T_TXT,             /* TXT query.  */
          0, 1,                 /* Class IN.  */
        };
      unsigned char response[512];
      int ret = res_send (query, sizeof (query), response, sizeof (response));

      char expected_name[20];
      /* The name is uncompressed in the query, so we can reference it
         directly.  */
      TEST_VERIFY_EXIT (ns_name_ntop (query + 12, expected_name,
                                      sizeof (expected_name)) >= 0);
      TEST_COMPARE (ret,
                    (ssize_t) sizeof (query)
                    + 2             /* Compression reference.  */
                    + 2 + 2 + 4 + 2 /* Type, class, TTL, RDATA length.  */
                    + 1             /* Pascal-style string length.  */
                    + strlen (expected_name));

      /* Mark as answer, with recursion available, and one answer.  */
      query[2] = 0x81;
      query[3] = 0x80;
      query[7] = 1;

      /* Prefix of the response must match the query.  */
      TEST_COMPARE (memcmp (response, query, sizeof (query)), 0);

      /* The actual answer follows, starting with the compression
         reference.  */
      unsigned char *p = response + sizeof (query);
      TEST_COMPARE (*p++, 0xc0);
      TEST_COMPARE (*p++, 0x0c);

      /* Type and class.  */
      TEST_COMPARE (*p++, 0);
      TEST_COMPARE (*p++, T_TXT);
      TEST_COMPARE (*p++, 0);
      TEST_COMPARE (*p++, C_IN);

      /* TTL.  */
      TEST_COMPARE (*p++, 0x12);
      TEST_COMPARE (*p++, 0x34);
      TEST_COMPARE (*p++, 0x56);
      TEST_COMPARE (*p++, 0x78);

      /* RDATA length.  */
      TEST_COMPARE (*p++, 0);
      TEST_COMPARE (*p++, 1 + strlen (expected_name));

      /* RDATA.  */
      TEST_COMPARE (*p++, strlen (expected_name));
      TEST_COMPARE (memcmp (p, expected_name, strlen (expected_name)), 0);
    }

  resolv_test_end (aux);

  return 0;
}

#include <support/test-driver.c>
