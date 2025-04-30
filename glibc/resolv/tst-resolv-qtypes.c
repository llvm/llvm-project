/* Exercise low-level query functions with different QTYPEs.
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

#include <resolv.h>
#include <string.h>
#include <support/check.h>
#include <support/check_nss.h>
#include <support/resolv_test.h>
#include <support/support.h>
#include <support/test-driver.h>
#include <support/xmemstream.h>

/* If ture, the response function will send the actual response packet
   over TCP instead of UDP.  */
static volatile bool force_tcp;

/* Send back a fake resource record matching the QTYPE.  */
static void
response (const struct resolv_response_context *ctx,
          struct resolv_response_builder *b,
          const char *qname, uint16_t qclass, uint16_t qtype)
{
  if (force_tcp && ctx->tcp)
    {
      resolv_response_init (b, (struct resolv_response_flags) { .tc = 1 });
      resolv_response_add_question (b, qname, qclass, qtype);
      return;
    }

  resolv_response_init (b, (struct resolv_response_flags) { });
  resolv_response_add_question (b, qname, qclass, qtype);
  resolv_response_section (b, ns_s_an);
  resolv_response_open_record (b, qname, qclass, qtype, 0);
  resolv_response_add_data (b, &qtype, sizeof (qtype));
  resolv_response_close_record (b);
}

static const char domain[] = "www.example.com";

static int
wrap_res_query (int type, unsigned char *answer, int answer_length)
{
  return res_query (domain, C_IN, type, answer, answer_length);
}

static int
wrap_res_search (int type, unsigned char *answer, int answer_length)
{
  return res_query (domain, C_IN, type, answer, answer_length);
}

static int
wrap_res_querydomain (int type, unsigned char *answer, int answer_length)
{
  return res_querydomain ("www", "example.com", C_IN, type,
                           answer, answer_length);
}

static int
wrap_res_send (int type, unsigned char *answer, int answer_length)
{
  unsigned char buf[512];
  int ret = res_mkquery (QUERY, domain, C_IN, type,
                         (const unsigned char *) "", 0, NULL,
                         buf, sizeof (buf));
  if (type < 0 || type >= 65536)
    {
      /* res_mkquery fails for out-of-range record types.  */
      TEST_VERIFY_EXIT (ret == -1);
      return -1;
    }
  TEST_VERIFY_EXIT (ret > 12);  /* DNS header length.  */
  return res_send (buf, ret, answer, answer_length);
}

static int
wrap_res_nquery (int type, unsigned char *answer, int answer_length)
{
  return res_nquery (&_res, domain, C_IN, type, answer, answer_length);
}

static int
wrap_res_nsearch (int type, unsigned char *answer, int answer_length)
{
  return res_nquery (&_res, domain, C_IN, type, answer, answer_length);
}

static int
wrap_res_nquerydomain (int type, unsigned char *answer, int answer_length)
{
  return res_nquerydomain (&_res, "www", "example.com", C_IN, type,
                           answer, answer_length);
}

static int
wrap_res_nsend (int type, unsigned char *answer, int answer_length)
{
  unsigned char buf[512];
  int ret = res_nmkquery (&_res, QUERY, domain, C_IN, type,
                         (const unsigned char *) "", 0, NULL,
                         buf, sizeof (buf));
  if (type < 0 || type >= 65536)
    {
      /* res_mkquery fails for out-of-range record types.  */
      TEST_VERIFY_EXIT (ret == -1);
      return -1;
    }
  TEST_VERIFY_EXIT (ret > 12);  /* DNS header length.  */
  return res_nsend (&_res, buf, ret, answer, answer_length);
}

static void
test_function (const char *fname,
               int (*func) (int type,
                            unsigned char *answer, int answer_length))
{
  unsigned char buf[512];
  for (int tcp = 0; tcp < 2; ++tcp)
    {
      force_tcp = tcp;
      for (unsigned int type = 1; type <= 65535; ++type)
        {
          if (test_verbose)
            printf ("info: sending QTYPE %d with %s (tcp=%d)\n",
                    type, fname, tcp);
          int ret = func (type, buf, sizeof (buf));
          if (ret != 47)
            FAIL_EXIT1 ("%s tcp=%d qtype=%d return value %d",
                        fname,tcp, type, ret);
          /* One question, one answer record.  */
          TEST_VERIFY (memcmp (buf + 4, "\0\1\0\1\0\0\0\0", 8) == 0);
          /* Question section.  */
          static const char qname[] = "\3www\7example\3com";
          size_t qname_length = sizeof (qname);
          TEST_VERIFY (memcmp (buf + 12, qname, qname_length) == 0);
          /* RDATA part of answer.  */
          uint16_t type16 = type;
          TEST_VERIFY (memcmp (buf + ret - 2, &type16, sizeof (type16)) == 0);
        }
    }

  TEST_VERIFY (func (-1, buf, sizeof (buf) == -1));
  TEST_VERIFY (func (65536, buf, sizeof (buf) == -1));
}

static int
do_test (void)
{
  struct resolv_redirect_config config =
    {
      .response_callback = response,
    };
  struct resolv_test *obj = resolv_test_start (config);

  test_function ("res_query", &wrap_res_query);
  test_function ("res_search", &wrap_res_search);
  test_function ("res_querydomain", &wrap_res_querydomain);
  test_function ("res_send", &wrap_res_send);

  test_function ("res_nquery", &wrap_res_nquery);
  test_function ("res_nsearch", &wrap_res_nsearch);
  test_function ("res_nquerydomain", &wrap_res_nquerydomain);
  test_function ("res_nsend", &wrap_res_nsend);

  resolv_test_end (obj);
  return 0;
}

#define TIMEOUT 300
#include <support/test-driver.c>
