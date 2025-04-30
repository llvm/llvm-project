/* Test __libc_res_nsend buffer mismanagement, basic TCP coverage.
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
#include <stdlib.h>
#include <string.h>
#include <support/check.h>
#include <support/check_nss.h>
#include <support/resolv_test.h>
#include <support/xthread.h>
#include <support/xmemstream.h>

static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

static int initial_address_count = 1;
static int subsequent_address_count = 2000;
static int response_number = 0;

static void
response (const struct resolv_response_context *ctx,
          struct resolv_response_builder *b,
          const char *qname, uint16_t qclass, uint16_t qtype)
{
  TEST_VERIFY_EXIT (qname != NULL);

  /* If not using TCP, just force its use.  */
  if (!ctx->tcp)
    {
      struct resolv_response_flags flags = {.tc = true};
      resolv_response_init (b, flags);
      resolv_response_add_question (b, qname, qclass, qtype);
      return;
    }

  struct resolv_response_flags flags = {};
  resolv_response_init (b, flags);
  resolv_response_add_question (b, qname, qclass, qtype);

  resolv_response_section (b, ns_s_an);

  /* The number of addresses (in the additional section) for the name
     server record (in the authoritative section).  */
  int address_count;
  xpthread_mutex_lock (&lock);
  ++response_number;
  if (response_number == 1)
    address_count = initial_address_count;
  else if (response_number == 2)
    {
      address_count = 0;
      resolv_response_drop (b);
      resolv_response_close (b);
    }
  else
    address_count = subsequent_address_count;
  xpthread_mutex_unlock (&lock);

  /* Only add the address record to the answer section if we requested
     any name server addresses.  */
  if (address_count > 0)
    {
      resolv_response_open_record (b, qname, qclass, qtype, 0);
      switch (qtype)
        {
        case T_A:
          {
            char ipv4[4] = {10, response_number >> 8, response_number, 0};
            ipv4[3] = 2 * ctx->tcp + 4 * ctx->server_index;
            resolv_response_add_data (b, &ipv4, sizeof (ipv4));
          }
          break;
        case T_AAAA:
          {
            char ipv6[16]
              = {0x20, 0x01, 0xd, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0,
                 response_number >> 8, response_number, 0, 0};
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

      /* Add the name server record.  */
      resolv_response_section (b, ns_s_ns);
      resolv_response_open_record (b, "example", C_IN, T_NS, 0);
      resolv_response_add_name (b, "ns.example");
      resolv_response_close_record (b);

      /* Increase the response size with name server addresses.  These
         addresses are not copied out of nss_dns, and thus do not
         trigger getaddrinfo retries with a larger buffer, making
         testing more predictable.  */
      resolv_response_section (b, ns_s_ar);
      for (int i = 1; i <= address_count; ++i)
        {
          resolv_response_open_record (b, "ns.example", qclass, qtype, 0);
          switch (qtype)
            {
            case T_A:
              {
                char ipv4[4] = {response_number, i >> 8, i, 0};
                ipv4[3] = 2 * ctx->tcp + 4 * ctx->server_index;
                resolv_response_add_data (b, &ipv4, sizeof (ipv4));
              }
              break;
            case T_AAAA:
              {
                char ipv6[16]
                  = {0x20, 0x01, 0xd, 0xb8, 0, 0, 0, 0, 0, 0,
                     response_number >> 8, response_number,
                     i >> 8, i, 0, 0};
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
}

static char *
expected_result (unsigned port, unsigned response_number)
{
  struct xmemstream mem;
  xopen_memstream (&mem);
  /* We fail the second TCP query to the first server by closing the
     connection immediately, without returning any data.  This should
     cause failover to the second server.  */
  int server_index = 1;
  fprintf (mem.out, "address: STREAM/TCP 10.%u.%u.%u %u\n",
           (response_number >> 8) & 0xff, response_number & 0xff,
           2 + 4 * server_index, port);
  fprintf (mem.out, "address: STREAM/TCP 2001:db8::%x:%x %u\n",
           (response_number + 1) & 0xffff,
           2 + 4 * server_index, port);
  xfclose_memstream (&mem);
  return mem.buffer;
}

static void
test_different_sizes (void)
{
  struct addrinfo hints =
    {
      .ai_family = AF_UNSPEC,
      .ai_socktype = SOCK_STREAM,
      .ai_protocol = IPPROTO_TCP,
    };
  struct addrinfo *ai;
  char *expected;
  int ret;

  /* This magic number produces a response size close to 2048
     bytes.  */
  initial_address_count = 124;
  response_number = 0;

  ret = getaddrinfo ("www.example", "80", &hints, &ai);
  expected = expected_result (80, 3);
  check_addrinfo ("www.example:80", ai, ret, expected);
  if (ret == 0)
    freeaddrinfo (ai);
  free (expected);

  response_number = 0;
  ret = getaddrinfo ("www123.example", "80", &hints, &ai);
  if (ret == 0)
    freeaddrinfo (ai);

  response_number = 0;
  ret = getaddrinfo ("www1234.example", "80", &hints, &ai);
  if (ret == 0)
    freeaddrinfo (ai);

  response_number = 0;
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
