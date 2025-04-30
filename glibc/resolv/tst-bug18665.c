/* Test for __libc_res_nsend buffer mismanagent (bug 18665), UDP case.
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

#include <errno.h>
#include <netdb.h>
#include <resolv.h>
#include <stdio.h>
#include <string.h>
#include <support/check.h>
#include <support/resolv_test.h>
#include <support/xthread.h>

static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

static int initial_address_count;
static int response_count;

static void
response (const struct resolv_response_context *ctx,
          struct resolv_response_builder *b,
          const char *qname, uint16_t qclass, uint16_t qtype)
{
  TEST_VERIFY_EXIT (qname != NULL);
  struct resolv_response_flags flags = {};
  resolv_response_init (b, flags);
  resolv_response_add_question (b, qname, qclass, qtype);

  resolv_response_section (b, ns_s_an);

  /* Add many A/AAAA records to the second response.  */
  int address_count;
  xpthread_mutex_lock (&lock);
  if (response_count == 0)
    address_count = initial_address_count;
  else
    address_count = 2000;
  ++response_count;
  xpthread_mutex_unlock (&lock);

  for (int i = 0; i < address_count; ++i)
    {
      resolv_response_open_record (b, qname, qclass, qtype, 0);
      switch (qtype)
        {
        case T_A:
          {
            char ipv4[4] = {10, i >> 8, i, 0};
            ipv4[3] = 2 * ctx->tcp + 4 * ctx->server_index;
            resolv_response_add_data (b, &ipv4, sizeof (ipv4));
          }
          break;
        case T_AAAA:
          {
            char ipv6[16]
              = {0x20, 0x01, 0xd, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 i >> 8, i, 0};
            ipv6[15] = 2 * ctx->tcp + 4 * ctx->server_index;
            resolv_response_add_data (b, &ipv6, sizeof (ipv6));
          }
          break;
        default:
          support_record_failure ();
          printf ("error: unexpected QTYPE: %s/%u/%u\n",
                  qname, qclass, qtype);
        }
      resolv_response_close_record (b);
    }
}

static void
test_different_sizes (void)
{
  struct addrinfo hints = { .ai_family = AF_UNSPEC, };
  struct addrinfo *ai;
  int ret;

  /* This magic number produces a response size close to 2048
     bytes.  */
  initial_address_count = 126;
  response_count = 0;

  ret = getaddrinfo ("www.example", "80", &hints, &ai);
  if (ret == 0)
    freeaddrinfo (ai);

  response_count = 0;
  ret = getaddrinfo ("www123.example", "80", &hints, &ai);
  if (ret == 0)
    freeaddrinfo (ai);

  response_count = 0;
  ret = getaddrinfo ("www1234.example", "80", &hints, &ai);
  if (ret == 0)
    freeaddrinfo (ai);

  response_count = 0;
  ret = getaddrinfo ("www12345.example", "80", &hints, &ai);
  if (ret == 0)
    freeaddrinfo (ai);
}

static int
do_test (void)
{
  struct resolv_test *obj = resolv_test_start
    ((struct resolv_redirect_config)
     {
       .response_callback = response
     });

  test_different_sizes ();

  _res.options |= RES_SNGLKUP;
  test_different_sizes ();

  _res.options |= RES_SNGLKUPREOP;
  test_different_sizes ();

  resolv_test_end (obj);
  return 0;
}

#include <support/test-driver.c>
