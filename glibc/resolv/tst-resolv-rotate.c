/* Check that RES_ROTATE works with few nameserver entries (bug 13028).
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

#include <netdb.h>
#include <resolv.h>
#include <stdlib.h>
#include <string.h>
#include <support/check.h>
#include <support/check_nss.h>
#include <support/resolv_test.h>
#include <support/test-driver.h>

static volatile int drop_server = -1;
static volatile unsigned int query_counts[resolv_max_test_servers];

static const char address_ipv4[4] = {192, 0, 2, 1};
static const char address_ipv6[16]
  = {0x20, 0x01, 0xd, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};

static void
response (const struct resolv_response_context *ctx,
          struct resolv_response_builder *b,
          const char *qname, uint16_t qclass, uint16_t qtype)
{
  if (ctx->server_index == drop_server)
    {
      resolv_response_drop (b);
      resolv_response_close (b);
      return;
    }

  bool force_tcp = strncmp (qname, "2.", 2) == 0;
  struct resolv_response_flags flags = {.tc = force_tcp && !ctx->tcp};
  resolv_response_init (b, flags);
  resolv_response_add_question (b, qname, qclass, qtype);
  if (flags.tc)
    return;

  TEST_VERIFY_EXIT (ctx->server_index < resolv_max_test_servers);
  ++query_counts[ctx->server_index];

  resolv_response_section (b, ns_s_an);
  resolv_response_open_record (b, qname, qclass, qtype, 0);
  switch (qtype)
    {
    case T_A:
      {
        char addr[sizeof (address_ipv4)];
        memcpy (addr, address_ipv4, sizeof (address_ipv4));
        addr[3] = 1 + ctx->tcp;
        resolv_response_add_data (b, addr, sizeof (addr));
      }
      break;
    case T_AAAA:
      {
        char addr[sizeof (address_ipv6)];
        memcpy (addr, address_ipv6, sizeof (address_ipv6));
        addr[15] = 1 + ctx->tcp;
        resolv_response_add_data (b, addr, sizeof (addr));
      }
      break;
    case T_PTR:
      if (force_tcp)
        resolv_response_add_name (b, "2.host.example");
      else
        resolv_response_add_name (b, "host.example");
      break;
    default:
      FAIL_EXIT1 ("unexpected QTYPE: %s/%u/%u", qname, qclass, qtype);
    }
  resolv_response_close_record (b);
}

static void
check_forward_1 (const char *name, int family)
{
  unsigned char lsb;
  if (strncmp (name, "2.", 2) == 0)
    lsb = 2;
  else
    lsb = 1;

  char expected_hostent_v4[200];
  snprintf (expected_hostent_v4, sizeof (expected_hostent_v4),
            "name: %s\naddress: 192.0.2.%d\n", name, lsb);
  char expected_hostent_v6[200];
  snprintf (expected_hostent_v6, sizeof (expected_hostent_v6),
            "name: %s\naddress: 2001:db8::%d\n", name, lsb);
  char expected_ai[200];

  unsigned char address[16];
  size_t address_length;

  char *expected_hostent;
  switch (family)
    {
    case AF_INET:
      expected_hostent = expected_hostent_v4;
      snprintf (expected_ai, sizeof (expected_ai),
                "address: STREAM/TCP 192.0.2.%d 80\n", lsb);
      TEST_VERIFY_EXIT (sizeof (address_ipv4) == sizeof (struct in_addr));
      memcpy (address, address_ipv4, sizeof (address_ipv4));
      address_length = sizeof (address_ipv4);
      break;
    case AF_INET6:
      expected_hostent = expected_hostent_v6;
      snprintf (expected_ai, sizeof (expected_ai),
                "address: STREAM/TCP 2001:db8::%d 80\n", lsb);
      TEST_VERIFY_EXIT (sizeof (address_ipv6) == sizeof (struct in6_addr));
      memcpy (address, address_ipv6, sizeof (address_ipv6));
      address_length = sizeof (address_ipv6);
      break;
    case AF_UNSPEC:
      expected_hostent = NULL;
      snprintf (expected_ai, sizeof (expected_ai),
                "address: STREAM/TCP 192.0.2.%d 80\n"
                "address: STREAM/TCP 2001:db8::%d 80\n",
                lsb, lsb);
      address_length = 0;
      break;
    default:
      FAIL_EXIT1 ("unknown address family %d", family);
    }


  if (family == AF_INET)
    {
      struct hostent *e = gethostbyname (name);
      check_hostent (name, e, expected_hostent_v4);
    }

  if (family != AF_UNSPEC)
    {
      struct hostent *e = gethostbyname2 (name, family);
      check_hostent (name, e, expected_hostent);
    }

  if (address_length > 0)
    {
      address[address_length - 1] = lsb;
      struct hostent *e = gethostbyaddr (address, address_length, family);
      check_hostent (name, e, expected_hostent);
    }

  struct addrinfo hints =
    {
      .ai_family = family,
      .ai_socktype = SOCK_STREAM,
      .ai_protocol = IPPROTO_TCP,
    };
  struct addrinfo *ai;
  int ret = getaddrinfo (name, "80", &hints, &ai);
  check_addrinfo (name, ai, ret, expected_ai);
  if (ret == 0)
    {
      for (struct addrinfo *p = ai; p != NULL; p = p->ai_next)
        {
          char host[200];
          ret = getnameinfo (p->ai_addr, p->ai_addrlen,
                             host, sizeof (host),
                             NULL, 0, /* service */
                             0);
          if (ret != 0)
            {
              support_record_failure ();
              printf ("error: getnameinfo: %d\n", ret);
            }
          else
            {
              if (lsb == 1)
                TEST_VERIFY (strcmp (host, "host.example") == 0);
              else
                TEST_VERIFY (strcmp (host, "2.host.example") == 0);
            }
        }
      freeaddrinfo (ai);
    }
}

static void
check_forward (int family)
{
  check_forward_1 ("host.example", family);
  check_forward_1 ("2.host.example", family);
}

static int
do_test (void)
{
  for (int force_tcp = 0; force_tcp < 2; ++force_tcp)
    for (int nscount = 1; nscount <= 3; ++nscount)
      for (int disable_server = -1; disable_server < nscount; ++disable_server)
        for (drop_server = -1; drop_server < nscount; ++drop_server)
          {
            /* A disabled server will never receive queries and
               therefore cannot drop them.  */
            if (drop_server >= 0 && drop_server == disable_server)
              continue;
            /* No servers remaining to query, all queries are expected
               to fail.  */
            int broken_servers = (disable_server >= 0) + (drop_server >= 0);
            if (nscount <= broken_servers)
              continue;

            if (test_verbose > 0)
              printf ("info: tcp=%d nscount=%d disable=%d drop=%d\n",
                      force_tcp, nscount, disable_server, drop_server);
            struct resolv_redirect_config config =
              {
                .response_callback = response,
                .nscount = nscount
              };
            if (disable_server >= 0)
              {
                config.servers[disable_server].disable_udp = true;
                config.servers[disable_server].disable_tcp = true;
              }

            struct resolv_test *aux = resolv_test_start (config);
            _res.options |= RES_ROTATE;

            /* Run a few queries to make sure that all of them
               succeed.  We always perform more than nscount queries,
               so we cover all active servers due to RES_ROTATE.  */
            for (size_t i = 0; i < resolv_max_test_servers; ++i)
              query_counts[i] = 0;
            check_forward (AF_INET);
            check_forward (AF_INET6);
            check_forward (AF_UNSPEC);

            for (int i = 0; i < nscount; ++i)
              {
                if (i != disable_server && i != drop_server
                    && query_counts[i] == 0)
                  {
                    support_record_failure ();
                    printf ("error: nscount=%d, but no query to server %d\n",
                            nscount, i);
                  }
              }

            resolv_test_end (aux);
          }
  return 0;
}

#define TIMEOUT 300
#include <support/test-driver.c>
