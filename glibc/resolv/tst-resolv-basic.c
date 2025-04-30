/* Test basic nss_dns functionality and the resolver test harness itself.
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <support/check.h>
#include <support/check_nss.h>
#include <support/format_nss.h>
#include <support/resolv_test.h>
#include <support/support.h>

#define LONG_NAME                                                       \
  "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaax."    \
  "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaay."    \
  "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaz."    \
  "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaat"

static void
response (const struct resolv_response_context *ctx,
          struct resolv_response_builder *b,
          const char *qname, uint16_t qclass, uint16_t qtype)
{
  TEST_VERIFY_EXIT (qname != NULL);

  /* The "t." prefix can be used to request TCP fallback.  */
  bool force_tcp;
  if (strncmp ("t.", qname, 2) == 0)
    force_tcp = true;
  else
    force_tcp = false;
  const char *qname_compare;
  if (force_tcp)
    qname_compare = qname + 2;
  else
    qname_compare = qname;
  enum {www, alias, nxdomain, long_name, nodata} requested_qname;
  if (strcmp (qname_compare, "www.example") == 0)
    requested_qname = www;
  else if (strcmp (qname_compare, "alias.example") == 0)
    requested_qname = alias;
  else if (strcmp (qname_compare, "nxdomain.example") == 0)
    requested_qname = nxdomain;
  else if (strcmp (qname_compare, LONG_NAME) == 0)
    requested_qname = long_name;
  else if (strcmp (qname_compare, "nodata.example") == 0)
    requested_qname = nodata;
  else
    {
      support_record_failure ();
      printf ("error: unexpected QNAME: %s\n", qname);
      return;
    }
  TEST_VERIFY_EXIT (qclass == C_IN);
  struct resolv_response_flags flags = {.tc = force_tcp && !ctx->tcp};
  if (requested_qname == nxdomain)
    flags.rcode = 3;            /* NXDOMAIN */
  resolv_response_init (b, flags);
  resolv_response_add_question (b, qname, qclass, qtype);
  if (requested_qname == nxdomain || flags.tc)
    return;

  resolv_response_section (b, ns_s_an);
  switch (requested_qname)
    {
    case www:
    case long_name:
      resolv_response_open_record (b, qname, qclass, qtype, 0);
      break;
    case alias:
      resolv_response_open_record (b, qname, qclass, T_CNAME, 0);
      resolv_response_add_name (b, "www.example");
      resolv_response_close_record (b);
      resolv_response_open_record (b, "www.example", qclass, qtype, 0);
      break;
    case nodata:
      return;
    case nxdomain:
      FAIL_EXIT1 ("unreachable");
    }
  switch (qtype)
    {
    case T_A:
      {
        char ipv4[4] = {192, 0, 2, 17};
        ipv4[3] += requested_qname + 2 * ctx->tcp + 4 * ctx->server_index;
        resolv_response_add_data (b, &ipv4, sizeof (ipv4));
      }
      break;
    case T_AAAA:
      {
        char ipv6[16]
          = {0x20, 0x01, 0xd, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
        ipv6[15] += requested_qname + 2 * ctx->tcp + 4 * ctx->server_index;
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

static void
check_h (const char *name, int family, const char *expected)
{
  if (family == AF_INET)
    {
      char *query = xasprintf ("gethostbyname (\"%s\")", name);
      check_hostent (query, gethostbyname (name), expected);
      free (query);
    }
  {
    char *query = xasprintf ("gethostbyname2 (\"%s\", %d)", name, family);
    check_hostent (query, gethostbyname2 (name, family), expected);
    free (query);
  }

  bool too_small = true;
  for (unsigned int offset = 0; offset < 8; ++offset)
    for (unsigned int size = 1; too_small; ++size)
      {
        char *buf = xmalloc (offset + size);
        too_small = false;

        struct hostent hostbuf;
        struct hostent *result;
        int herror;
        if (family == AF_INET)
          {
            char *query = xasprintf ("gethostbyname (\"%s\") %u/%u",
                                     name, offset, size);
            int ret = gethostbyname_r
              (name, &hostbuf, buf + offset, size, &result, &herror);
            if (ret == 0)
              {
                h_errno = herror;
                check_hostent (query, result, expected);
              }
            else if (ret == ERANGE)
              too_small = true;
            else
              {
                errno = ret;
                FAIL_EXIT1 ("gethostbyname_r: %m");
              }
            free (query);
            memset (buf, 0, offset + size);
          }
        char *query = xasprintf ("gethostbyname2 (\"%s\", %d) %u/%u",
                                 name, family, offset, size);
        int ret = gethostbyname2_r
          (name, family, &hostbuf, buf + offset, size, &result, &herror);
        if (ret == 0)
          {
            h_errno = herror;
            check_hostent (query, result, expected);
          }
        else if (ret == ERANGE)
          too_small = true;
        else
          {
            errno = ret;
            FAIL_EXIT1 ("gethostbyname_r: %m");
          }
        free (buf);
        free (query);
      }
}

static void
check_ai_hints (const char *name, const char *service,
                struct addrinfo hints, const char *expected)
{
  struct addrinfo *ai;
  char *query = xasprintf ("%s:%s [%d]/0x%x", name, service,
                           hints.ai_family, hints.ai_flags);
  int ret = getaddrinfo (name, service, &hints, &ai);
  check_addrinfo (query, ai, ret, expected);
  if (ret == 0)
    freeaddrinfo (ai);
  free (query);
}

static void
check_ai (const char *name, const char *service,
          int family, const char *expected)
{
  return check_ai_hints (name, service,
                         (struct addrinfo) { .ai_family = family, },
                         expected);
}

/* Test for bug 21295: getaddrinfo used to discard address information
   instead of merging it.  */
static void
test_bug_21295 (void)
{
  /* The address order is unpredictable.  There are two factors which
     contribute to that: The stub resolver does not perform proper
     response matching for A/AAAA queries (an A response could be
     associated with an AAAA query and vice versa), and without
     namespaces, system configuration could affect address
     ordering.  */
  for (int do_tcp = 0; do_tcp < 2; ++do_tcp)
    {
      const struct addrinfo hints =
        {
          .ai_family = AF_INET6,
          .ai_socktype = SOCK_STREAM,
          .ai_flags = AI_V4MAPPED | AI_ALL,
        };
      const char *qname;
      if (do_tcp)
        qname = "t.www.example";
      else
        qname = "www.example";
      struct addrinfo *ai = NULL;
      int ret = getaddrinfo (qname, "80", &hints, &ai);
      TEST_VERIFY_EXIT (ret == 0);

      const char *expected_a;
      const char *expected_b;
      if (do_tcp)
        {
          expected_a = "flags: AI_V4MAPPED AI_ALL\n"
            "address: STREAM/TCP 2001:db8::3 80\n"
            "address: STREAM/TCP ::ffff:192.0.2.19 80\n";
          expected_b = "flags: AI_V4MAPPED AI_ALL\n"
            "address: STREAM/TCP ::ffff:192.0.2.19 80\n"
            "address: STREAM/TCP 2001:db8::3 80\n";
        }
      else
        {
          expected_a = "flags: AI_V4MAPPED AI_ALL\n"
            "address: STREAM/TCP 2001:db8::1 80\n"
            "address: STREAM/TCP ::ffff:192.0.2.17 80\n";
          expected_b = "flags: AI_V4MAPPED AI_ALL\n"
            "address: STREAM/TCP ::ffff:192.0.2.17 80\n"
            "address: STREAM/TCP 2001:db8::1 80\n";
        }

      char *actual = support_format_addrinfo (ai, ret);
      if (!(strcmp (actual, expected_a) == 0
            || strcmp (actual, expected_b) == 0))
        {
          support_record_failure ();
          printf ("error: %s: unexpected response (TCP: %d):\n%s\n",
                  __func__, do_tcp, actual);
        }
      free (actual);
      freeaddrinfo (ai);
    }
}

/* Run tests which do not expect any data.  */
static void
test_nodata_nxdomain (void)
{
  /* Iterate through different address families.  */
  int families[] = { AF_UNSPEC, AF_INET, AF_INET6, -1 };
  for (int i = 0; families[i] >= 0; ++i)
    /* If do_tcp, prepend "t." to the name to trigger TCP
       fallback.  */
    for (int do_tcp = 0; do_tcp < 2; ++do_tcp)
      /* If do_nxdomain, trigger an NXDOMAIN error (DNS failure),
         otherwise use a NODATA response (empty but successful
         answer).  */
      for (int do_nxdomain = 0; do_nxdomain < 2; ++do_nxdomain)
        {
          int family = families[i];
          char *name = xasprintf ("%s%s.example",
                                  do_tcp ? "t." : "",
                                  do_nxdomain ? "nxdomain" : "nodata");

          if (family != AF_UNSPEC)
            {
              if (do_nxdomain)
                check_h (name, family, "error: HOST_NOT_FOUND\n");
              else
                check_h (name, family, "error: NO_ADDRESS\n");
            }

          const char *expected;
          if (do_nxdomain)
            expected = "error: Name or service not known\n";
          else
            expected = "error: No address associated with hostname\n";

          check_ai (name, "80", family, expected);

          struct addrinfo hints =
            {
              .ai_family = family,
              .ai_flags = AI_V4MAPPED | AI_ALL,
            };
          check_ai_hints (name, "80", hints, expected);
          hints.ai_flags |= AI_CANONNAME;
          check_ai_hints (name, "80", hints, expected);

          free (name);
        }
}

static int
do_test (void)
{
  struct resolv_test *aux = resolv_test_start
    ((struct resolv_redirect_config)
     {
       .response_callback = response,
     });

  check_h ("www.example", AF_INET,
           "name: www.example\n"
           "address: 192.0.2.17\n");
  check_h ("alias.example", AF_INET,
           "name: www.example\n"
           "alias: alias.example\n"
           "address: 192.0.2.18\n");
  check_h ("www.example", AF_INET6,
           "name: www.example\n"
           "address: 2001:db8::1\n");
  check_h ("alias.example", AF_INET6,
           "name: www.example\n"
           "alias: alias.example\n"
           "address: 2001:db8::2\n");
  check_h (LONG_NAME, AF_INET,
           "name: " LONG_NAME "\n"
           "address: 192.0.2.20\n");

  check_ai ("www.example", "80", AF_UNSPEC,
            "address: STREAM/TCP 192.0.2.17 80\n"
            "address: DGRAM/UDP 192.0.2.17 80\n"
            "address: RAW/IP 192.0.2.17 80\n"
            "address: STREAM/TCP 2001:db8::1 80\n"
            "address: DGRAM/UDP 2001:db8::1 80\n"
            "address: RAW/IP 2001:db8::1 80\n");
  check_ai_hints ("www.example", "80",
                  (struct addrinfo) { .ai_family = AF_UNSPEC,
                      .ai_flags = AI_CANONNAME, },
                  "flags: AI_CANONNAME\n"
                  "canonname: www.example\n"
                  "address: STREAM/TCP 192.0.2.17 80\n"
                  "address: DGRAM/UDP 192.0.2.17 80\n"
                  "address: RAW/IP 192.0.2.17 80\n"
                  "address: STREAM/TCP 2001:db8::1 80\n"
                  "address: DGRAM/UDP 2001:db8::1 80\n"
                  "address: RAW/IP 2001:db8::1 80\n");
  check_ai ("alias.example", "80", AF_UNSPEC,
            "address: STREAM/TCP 192.0.2.18 80\n"
            "address: DGRAM/UDP 192.0.2.18 80\n"
            "address: RAW/IP 192.0.2.18 80\n"
            "address: STREAM/TCP 2001:db8::2 80\n"
            "address: DGRAM/UDP 2001:db8::2 80\n"
            "address: RAW/IP 2001:db8::2 80\n");
  check_ai_hints ("alias.example", "80",
                  (struct addrinfo) { .ai_family = AF_UNSPEC,
                      .ai_flags = AI_CANONNAME, },
                  "flags: AI_CANONNAME\n"
                  "canonname: www.example\n"
                  "address: STREAM/TCP 192.0.2.18 80\n"
                  "address: DGRAM/UDP 192.0.2.18 80\n"
                  "address: RAW/IP 192.0.2.18 80\n"
                  "address: STREAM/TCP 2001:db8::2 80\n"
                  "address: DGRAM/UDP 2001:db8::2 80\n"
                  "address: RAW/IP 2001:db8::2 80\n");
  check_ai (LONG_NAME, "80", AF_UNSPEC,
            "address: STREAM/TCP 192.0.2.20 80\n"
            "address: DGRAM/UDP 192.0.2.20 80\n"
            "address: RAW/IP 192.0.2.20 80\n"
            "address: STREAM/TCP 2001:db8::4 80\n"
            "address: DGRAM/UDP 2001:db8::4 80\n"
            "address: RAW/IP 2001:db8::4 80\n");
  check_ai ("www.example", "80", AF_INET,
            "address: STREAM/TCP 192.0.2.17 80\n"
            "address: DGRAM/UDP 192.0.2.17 80\n"
            "address: RAW/IP 192.0.2.17 80\n");
  check_ai_hints ("www.example", "80",
                  (struct addrinfo) { .ai_family = AF_INET,
                      .ai_flags = AI_CANONNAME, },
                  "flags: AI_CANONNAME\n"
                  "canonname: www.example\n"
                  "address: STREAM/TCP 192.0.2.17 80\n"
                  "address: DGRAM/UDP 192.0.2.17 80\n"
                  "address: RAW/IP 192.0.2.17 80\n");
  check_ai ("alias.example", "80", AF_INET,
            "address: STREAM/TCP 192.0.2.18 80\n"
            "address: DGRAM/UDP 192.0.2.18 80\n"
            "address: RAW/IP 192.0.2.18 80\n");
  check_ai_hints ("alias.example", "80",
                  (struct addrinfo) { .ai_family = AF_INET,
                      .ai_flags = AI_CANONNAME, },
                  "flags: AI_CANONNAME\n"
                  "canonname: www.example\n"
                  "address: STREAM/TCP 192.0.2.18 80\n"
                  "address: DGRAM/UDP 192.0.2.18 80\n"
                  "address: RAW/IP 192.0.2.18 80\n");
  check_ai (LONG_NAME, "80", AF_INET,
            "address: STREAM/TCP 192.0.2.20 80\n"
            "address: DGRAM/UDP 192.0.2.20 80\n"
            "address: RAW/IP 192.0.2.20 80\n");
  check_ai ("www.example", "80", AF_INET6,
            "address: STREAM/TCP 2001:db8::1 80\n"
            "address: DGRAM/UDP 2001:db8::1 80\n"
            "address: RAW/IP 2001:db8::1 80\n");
  check_ai_hints ("www.example", "80",
                  (struct addrinfo) { .ai_family = AF_INET6,
                      .ai_flags = AI_CANONNAME, },
                  "flags: AI_CANONNAME\n"
                  "canonname: www.example\n"
                  "address: STREAM/TCP 2001:db8::1 80\n"
                  "address: DGRAM/UDP 2001:db8::1 80\n"
                  "address: RAW/IP 2001:db8::1 80\n");
  check_ai ("alias.example", "80", AF_INET6,
            "address: STREAM/TCP 2001:db8::2 80\n"
            "address: DGRAM/UDP 2001:db8::2 80\n"
            "address: RAW/IP 2001:db8::2 80\n");
  check_ai_hints ("alias.example", "80",
                  (struct addrinfo) { .ai_family = AF_INET6,
                      .ai_flags = AI_CANONNAME, },
                  "flags: AI_CANONNAME\n"
                  "canonname: www.example\n"
                  "address: STREAM/TCP 2001:db8::2 80\n"
                  "address: DGRAM/UDP 2001:db8::2 80\n"
                  "address: RAW/IP 2001:db8::2 80\n");
  check_ai (LONG_NAME, "80", AF_INET6,
            "address: STREAM/TCP 2001:db8::4 80\n"
            "address: DGRAM/UDP 2001:db8::4 80\n"
            "address: RAW/IP 2001:db8::4 80\n");

  check_h ("t.www.example", AF_INET,
           "name: t.www.example\n"
           "address: 192.0.2.19\n");
  check_h ("t.alias.example", AF_INET,
           "name: www.example\n"
           "alias: t.alias.example\n"
           "address: 192.0.2.20\n");
  check_h ("t.www.example", AF_INET6,
           "name: t.www.example\n"
           "address: 2001:db8::3\n");
  check_h ("t.alias.example", AF_INET6,
           "name: www.example\n"
           "alias: t.alias.example\n"
           "address: 2001:db8::4\n");
  check_ai ("t.www.example", "80", AF_UNSPEC,
            "address: STREAM/TCP 192.0.2.19 80\n"
            "address: DGRAM/UDP 192.0.2.19 80\n"
            "address: RAW/IP 192.0.2.19 80\n"
            "address: STREAM/TCP 2001:db8::3 80\n"
            "address: DGRAM/UDP 2001:db8::3 80\n"
            "address: RAW/IP 2001:db8::3 80\n");
  check_ai ("t.alias.example", "80", AF_UNSPEC,
            "address: STREAM/TCP 192.0.2.20 80\n"
            "address: DGRAM/UDP 192.0.2.20 80\n"
            "address: RAW/IP 192.0.2.20 80\n"
            "address: STREAM/TCP 2001:db8::4 80\n"
            "address: DGRAM/UDP 2001:db8::4 80\n"
            "address: RAW/IP 2001:db8::4 80\n");
  check_ai ("t.www.example", "80", AF_INET,
            "address: STREAM/TCP 192.0.2.19 80\n"
            "address: DGRAM/UDP 192.0.2.19 80\n"
            "address: RAW/IP 192.0.2.19 80\n");
  check_ai ("t.alias.example", "80", AF_INET,
            "address: STREAM/TCP 192.0.2.20 80\n"
            "address: DGRAM/UDP 192.0.2.20 80\n"
            "address: RAW/IP 192.0.2.20 80\n");
  check_ai ("t.www.example", "80", AF_INET6,
            "address: STREAM/TCP 2001:db8::3 80\n"
            "address: DGRAM/UDP 2001:db8::3 80\n"
            "address: RAW/IP 2001:db8::3 80\n");
  check_ai ("t.alias.example", "80", AF_INET6,
            "address: STREAM/TCP 2001:db8::4 80\n"
            "address: DGRAM/UDP 2001:db8::4 80\n"
            "address: RAW/IP 2001:db8::4 80\n");

  test_bug_21295 ();
  test_nodata_nxdomain ();

  resolv_test_end (aux);

  return 0;
}

#include <support/test-driver.c>
