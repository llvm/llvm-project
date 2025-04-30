/* Test parallel queries with transaction ID collisions.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#include <arpa/nameser.h>
#include <array_length.h>
#include <resolv-internal.h>
#include <resolv_context.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <support/check.h>
#include <support/check_nss.h>
#include <support/resolv_test.h>
#include <support/support.h>
#include <support/test-driver.h>

/* Result of parsing a DNS question name.

   A question name has the form reorder-N-M-rcode-C.example.net, where
   N and M are either 0 and 1, corresponding to the reorder member,
   and C is a number that will be stored in the rcode field.

   Also see parse_qname below.  */
struct parsed_qname
{
  /* The DNS response code requested from the first server.  The
     second server always responds with RCODE zero.  */
  int rcode;

  /* Indicates whether to perform reordering in the responses from the
     respective server.  */
  bool reorder[2];
};

/* Fills *PARSED based on QNAME.  */
static void
parse_qname (struct parsed_qname *parsed, const char *qname)
{
  int reorder0;
  int reorder1;
  int rcode;
  char *suffix;
  if (sscanf (qname, "reorder-%d-%d.rcode-%d.%ms",
              &reorder0, &reorder1, &rcode, &suffix) == 4)
    {
      if (reorder0 != 0)
        TEST_COMPARE (reorder0, 1);
      if (reorder1 != 0)
        TEST_COMPARE (reorder1, 1);
      TEST_VERIFY (rcode >= 0 && rcode <= 15);
      TEST_COMPARE_STRING (suffix, "example.net");
      free (suffix);

      parsed->rcode = rcode;
      parsed->reorder[0] = reorder0;
      parsed->reorder[1] = reorder1;
    }
  else
    FAIL_EXIT1 ("unexpected query: %s", qname);
}

/* Used to construct a response. The first server responds with an
   error, the second server succeeds.  */
static void
build_response (const struct resolv_response_context *ctx,
                struct resolv_response_builder *b,
                const char *qname, uint16_t qclass, uint16_t qtype)
{
  struct parsed_qname parsed;
  parse_qname (&parsed, qname);

  switch (ctx->server_index)
    {
    case 0:
      {
        struct resolv_response_flags flags = { 0 };
        if (parsed.rcode == 0)
          /* Simulate a delegation in case a NODATA (RCODE zero)
             response is requested.  */
          flags.clear_ra = true;
        else
          flags.rcode = parsed.rcode;

        resolv_response_init (b, flags);
        resolv_response_add_question (b, qname, qclass, qtype);
      }
      break;

    case 1:
      {
        struct resolv_response_flags flags = { 0, };
        resolv_response_init (b, flags);
        resolv_response_add_question (b, qname, qclass, qtype);

        resolv_response_section (b, ns_s_an);
        resolv_response_open_record (b, qname, qclass, qtype, 0);
        if (qtype == T_A)
          {
            char ipv4[4] = { 192, 0, 2, 1 };
            resolv_response_add_data (b, &ipv4, sizeof (ipv4));
          }
        else
          {
            char ipv6[16]
              = { 0x20, 0x01, 0xd, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };
            resolv_response_add_data (b, &ipv6, sizeof (ipv6));
          }
        resolv_response_close_record (b);
      }
      break;
    }
}

/* Used to reorder responses.  */
struct resolv_response_context *previous_query;

/* Used to keep track of the queries received.  */
static int previous_server_index = -1;
static uint16_t previous_qtype;

/* For each server, buffer the first query and then send both answers
   to the second query, reordered if requested.  */
static void
response (const struct resolv_response_context *ctx,
          struct resolv_response_builder *b,
          const char *qname, uint16_t qclass, uint16_t qtype)
{
  TEST_VERIFY (qtype == T_A || qtype == T_AAAA);
  if (ctx->server_index != 0)
    TEST_COMPARE (ctx->server_index, 1);

  struct parsed_qname parsed;
  parse_qname (&parsed, qname);

  if (previous_query == NULL)
    {
      /* No buffered query.  Record this query and do not send a
         response.  */
      TEST_COMPARE (previous_qtype, 0);
      previous_query = resolv_response_context_duplicate (ctx);
      previous_qtype = qtype;
      resolv_response_drop (b);
      previous_server_index = ctx->server_index;

      if (test_verbose)
        printf ("info: buffering first query for: %s\n", qname);
    }
  else
    {
      TEST_VERIFY (previous_query != 0);
      TEST_COMPARE (ctx->server_index, previous_server_index);
      TEST_VERIFY (previous_qtype != qtype); /* Not a duplicate.  */

      /* If reordering, send a response for this query explicitly, and
         then skip the implicit send.  */
      if (parsed.reorder[ctx->server_index])
        {
          if (test_verbose)
            printf ("info: sending reordered second response for: %s\n",
                    qname);
          build_response (ctx, b, qname, qclass, qtype);
          resolv_response_send_udp (ctx, b);
          resolv_response_drop (b);
        }

      /* Build a response for the previous query and send it, thus
         reordering the two responses.  */
      {
        if (test_verbose)
          printf ("info: sending first response for: %s\n", qname);
        struct resolv_response_builder *btmp
          = resolv_response_builder_allocate (previous_query->query_buffer,
                                              previous_query->query_length);
        build_response (ctx, btmp, qname, qclass, previous_qtype);
        resolv_response_send_udp (ctx, btmp);
        resolv_response_builder_free (btmp);
      }

      /* If not reordering, send the reply as usual.  */
      if (!parsed.reorder[ctx->server_index])
        {
          if (test_verbose)
            printf ("info: sending non-reordered second response for: %s\n",
                    qname);
          build_response (ctx, b, qname, qclass, qtype);
        }

      /* Unbuffer the response and prepare for the next query.  */
      resolv_response_context_free (previous_query);
      previous_query = NULL;
      previous_qtype = 0;
      previous_server_index = -1;
    }
}

/* Runs a query for QNAME and checks for the expected reply.  See
   struct parsed_qname for the expected format for QNAME.  */
static void
test_qname (const char *qname, int rcode)
{
  struct resolv_context *ctx = __resolv_context_get ();
  TEST_VERIFY_EXIT (ctx != NULL);

  unsigned char q1[512];
  int q1len = res_mkquery (QUERY, qname, C_IN, T_A, NULL, 0, NULL,
                           q1, sizeof (q1));
  TEST_VERIFY_EXIT (q1len > 12);

  unsigned char q2[512];
  int q2len = res_mkquery (QUERY, qname, C_IN, T_AAAA, NULL, 0, NULL,
                           q2, sizeof (q2));
  TEST_VERIFY_EXIT (q2len > 12);

  /* Produce a transaction ID collision.  */
  memcpy (q2, q1, 2);

  unsigned char ans1[512];
  unsigned char *ans1p = ans1;
  unsigned char *ans2p = NULL;
  int nans2p = 0;
  int resplen2 = 0;
  int ans2p_malloced = 0;

  /* Perform a parallel A/AAAA query.  */
  int resplen1 = __res_context_send (ctx, q1, q1len, q2, q2len,
                                     ans1, sizeof (ans1), &ans1p,
                                     &ans2p, &nans2p,
                                     &resplen2, &ans2p_malloced);

  TEST_VERIFY (resplen1 > 12);
  TEST_VERIFY (resplen2 > 12);
  if (resplen1 <= 12 || resplen2 <= 12)
    return;

  if (rcode == 1 || rcode == 3)
    {
      /* Format Error and Name Error responses does not trigger
         switching to the next server.  */
      TEST_COMPARE (ans1p[3] & 0x0f, rcode);
      TEST_COMPARE (ans2p[3] & 0x0f, rcode);
      return;
    }

  /* The response should be successful.  */
  TEST_COMPARE (ans1p[3] & 0x0f, 0);
  TEST_COMPARE (ans2p[3] & 0x0f, 0);

  /* Due to bug 19691, the answer may not be in the slot matching the
     query.  Assume that the AAAA response is the longer one.  */
  unsigned char *a_answer;
  int a_answer_length;
  unsigned char *aaaa_answer;
  int aaaa_answer_length;
  if (resplen2 > resplen1)
    {
      a_answer = ans1p;
      a_answer_length = resplen1;
      aaaa_answer = ans2p;
      aaaa_answer_length = resplen2;
    }
  else
    {
      a_answer = ans2p;
      a_answer_length = resplen2;
      aaaa_answer = ans1p;
      aaaa_answer_length = resplen1;
    }

  {
    char *expected = xasprintf ("name: %s\n"
                                "address: 192.0.2.1\n",
                                qname);
    check_dns_packet (qname, a_answer, a_answer_length, expected);
    free (expected);
  }
  {
    char *expected = xasprintf ("name: %s\n"
                                "address: 2001:db8::1\n",
                                qname);
    check_dns_packet (qname, aaaa_answer, aaaa_answer_length, expected);
    free (expected);
  }

  if (ans2p_malloced)
    free (ans2p);

  __resolv_context_put (ctx);
}

static int
do_test (void)
{
  struct resolv_test *aux = resolv_test_start
    ((struct resolv_redirect_config)
     {
       .response_callback = response,

       /* The response callback use global state (the previous_*
          variables), and query processing must therefore be
          serialized.  */
       .single_thread_udp = true,
     });

  for (int rcode = 0; rcode <= 5; ++rcode)
    for (int do_reorder_0 = 0; do_reorder_0 < 2; ++do_reorder_0)
      for (int do_reorder_1 = 0; do_reorder_1 < 2; ++do_reorder_1)
        {
          char *qname = xasprintf ("reorder-%d-%d.rcode-%d.example.net",
                                   do_reorder_0, do_reorder_1, rcode);
          test_qname (qname, rcode);
          free (qname);
        }

  resolv_test_end (aux);

  return 0;
}

#include <support/test-driver.c>
