/* Test search/default domain name behavior.
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
#include <stdlib.h>
#include <string.h>
#include <support/check.h>
#include <support/check_nss.h>
#include <support/resolv_test.h>
#include <support/support.h>
#include <support/xmemstream.h>

struct item
{
  const char *name;
  int response;
};

const struct item items[] =
  {
    {"hostname.usersys.example.com", 1},
    {"hostname.corp.example.com", 1},
    {"hostname.example.com", 1},

    {"mail.corp.example.com", 1},
    {"mail.example.com", 1},

    {"file.corp.example.com", 2},
    {"file.corp", 1},
    {"file.example.com", 1},
    {"servfail-usersys.usersys.example.com", -ns_r_servfail},
    {"servfail-usersys.corp.example.com", 1},
    {"servfail-usersys.example.com", 1},
    {"servfail-corp.usersys.example.com", 1},
    {"servfail-corp.corp.example.com", -ns_r_servfail},
    {"servfail-corp.example.com", 1},
    {"www.example.com", 1},
    {"large.example.com", 200},

    /* Test query amplification with a SERVFAIL response combined with
       a large RRset.  */
    {"large-servfail.usersys.example.com", -ns_r_servfail},
    {"large-servfail.example.com", 2000},
    {}
  };

enum
  {
    name_not_found = -1,
    name_no_data = -2
  };

static int
find_name (const char *name)
{
  for (int i = 0; items[i].name != NULL; ++i)
    {
      if (strcmp (name, items[i].name) == 0)
        return i;
    }
  if (strcmp (name, "example.com") == 0
      || strcmp (name, "usersys.example.com") == 0
      || strcmp (name, "corp.example.com") == 0)
    return name_no_data;
  return name_not_found;
}

static int rcode_override_server_index = -1;
static int rcode_override;

static void
response (const struct resolv_response_context *ctx,
          struct resolv_response_builder *b,
          const char *qname, uint16_t qclass, uint16_t qtype)
{
  if (ctx->server_index == rcode_override_server_index)
    {
      struct resolv_response_flags flags = {.rcode = rcode_override};
      resolv_response_init (b, flags);
      resolv_response_add_question (b, qname, qclass, qtype);
      return;
    }

  int index = find_name (qname);
  struct resolv_response_flags flags = {};
  if (index == name_not_found)
    flags.rcode = ns_r_nxdomain;
  else if (index >= 0 && items[index].response < 0)
    flags.rcode = -items[index].response;
  else if (index >= 0 && items[index].response > 5 && !ctx->tcp)
    /* Force TCP if more than 5 addresses where requested.  */
    flags.tc = true;
  resolv_response_init (b, flags);
  resolv_response_add_question (b, qname, qclass, qtype);

  if (flags.tc || index < 0 || items[index].response < 0)
    return;

  resolv_response_section (b, ns_s_an);

  for (int i = 0; i < items[index].response; ++i)
    {
      resolv_response_open_record (b, qname, qclass, qtype, 0);

      switch (qtype)
        {
        case T_A:
          {
            char addr[4] = {10, index, i >> 8, i};
            resolv_response_add_data (b, addr, sizeof (addr));
          }
          break;
        case T_AAAA:
          {
            char addr[16]
              = {0x20, 0x01, 0xd, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, index + 1, (i + 1) >> 8, i + 1};
            resolv_response_add_data (b, addr, sizeof (addr));
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

enum output_format
  {
    format_get, format_gai
  };

static void
format_expected_1 (FILE *out, int family, enum output_format format, int index)
{
  for (int i = 0; i < items[index].response; ++i)
    {
      char address[200];
      switch (family)
        {
        case AF_INET:
          snprintf (address, sizeof (address), "10.%d.%d.%d",
                    index, (i >> 8) & 0xff, i & 0xff);
          break;
        case AF_INET6:
          snprintf (address, sizeof (address), "2001:db8::%x:%x",
                    index + 1, i + 1);
          break;
        default:
          FAIL_EXIT1 ("unreachable");
        }

      switch (format)
        {
        case format_get:
          fprintf (out, "address: %s\n", address);
          break;
        case format_gai:
          fprintf (out, "address: STREAM/TCP %s 80\n", address);
        }
    }
}

static char *
format_expected (const char *fqdn, int family, enum output_format format)
{
  int index = find_name (fqdn);
  TEST_VERIFY_EXIT (index >= 0);
  struct xmemstream stream;
  xopen_memstream (&stream);

  TEST_VERIFY_EXIT (items[index].response >= 0);
  if (format == format_get)
    fprintf (stream.out, "name: %s\n", items[index].name);
  if (family == AF_INET || family == AF_UNSPEC)
    format_expected_1 (stream.out, AF_INET, format, index);
  if (family == AF_INET6 || family == AF_UNSPEC)
    format_expected_1 (stream.out, AF_INET6, format, index);

  xfclose_memstream (&stream);
  return stream.buffer;
}

static void
do_get (const char *name, const char *fqdn, int family)
{
  char *expected = format_expected (fqdn, family, format_get);
  if (family == AF_INET)
    {
      char *query = xasprintf ("gethostbyname (\"%s\")", name);
      check_hostent (query, gethostbyname (name), expected);
      free (query);
    }
  char *query = xasprintf ("gethostbyname2 (\"%s\", %d)", name, family);
  check_hostent (query, gethostbyname2 (name, family), expected);

  /* Test res_search.  */
  int qtype;
  switch (family)
    {
    case AF_INET:
      qtype = T_A;
      break;
    case AF_INET6:
      qtype = T_AAAA;
      break;
    default:
      qtype = -1;
    }
  if (qtype >= 0)
    {
      int sz = 512;
      unsigned char *response = xmalloc (sz);
      int ret = res_search (name, C_IN, qtype, response, sz);
      TEST_VERIFY_EXIT (ret >= 0);
      if (ret > sz)
        {
          /* Truncation.  Retry with a larger buffer.  */
          sz = 65535;
          unsigned char *newptr = xrealloc (response, sz);
          response = newptr;

          ret = res_search (name, C_IN, qtype, response, sz);
          TEST_VERIFY_EXIT (ret >= 0);
          TEST_VERIFY_EXIT (ret < sz);
        }
      check_dns_packet (query, response, ret, expected);
      free (response);
    }

  free (query);
  free (expected);
}

static void
do_gai (const char *name, const char *fqdn, int family)
{
  struct addrinfo hints =
    {
      .ai_family = family,
      .ai_protocol = IPPROTO_TCP,
      .ai_socktype = SOCK_STREAM
    };
  struct addrinfo *ai;
  char *query = xasprintf ("%s:80 [%d]", name, family);
  int ret = getaddrinfo (name, "80", &hints, &ai);
  char *expected = format_expected (fqdn, family, format_gai);
  check_addrinfo (query, ai, ret, expected);
  if (ret == 0)
    freeaddrinfo (ai);
  free (expected);
  free (query);
}

static void
do_both (const char *name, const char *fqdn)
{
  do_get (name, fqdn, AF_INET);
  do_get (name, fqdn, AF_INET6);
  do_gai (name, fqdn, AF_INET);
  do_gai (name, fqdn, AF_INET6);
  do_gai (name, fqdn, AF_UNSPEC);
}

static void
do_test_all (bool unconnectable_server)
{
  struct resolv_redirect_config config =
    {
      .response_callback = response,
      .search = {"usersys.example.com", "corp.example.com", "example.com"},
    };
  struct resolv_test *obj = resolv_test_start (config);

  if (unconnectable_server)
    {
      /* 255.255.255.255 results in an immediate connect failure.  The
         next server will supply the answer instead.  This is a
         triggering condition for bug 19791.  */
      _res.nsaddr_list[0].sin_addr.s_addr = -1;
      _res.nsaddr_list[0].sin_port = htons (53);
    }

  do_both ("file", "file.corp.example.com");
  do_both ("www", "www.example.com");
  do_both ("servfail-usersys", "servfail-usersys.corp.example.com");
  do_both ("servfail-corp", "servfail-corp.usersys.example.com");
  do_both ("large", "large.example.com");
  do_both ("large-servfail", "large-servfail.example.com");
  do_both ("file.corp", "file.corp");

  /* Check that SERVFAIL and REFUSED responses do not alter the search
     path resolution.  */
  rcode_override_server_index = 0;
  rcode_override = ns_r_servfail;
  do_both ("hostname", "hostname.usersys.example.com");
  do_both ("large", "large.example.com");
  do_both ("large-servfail", "large-servfail.example.com");
  rcode_override = ns_r_refused;
  do_both ("hostname", "hostname.usersys.example.com");
  do_both ("large", "large.example.com");
  do_both ("large-servfail", "large-servfail.example.com");
  /* Likewise, but with an NXDOMAIN for the first search path
     entry.  */
  rcode_override = ns_r_servfail;
  do_both ("mail", "mail.corp.example.com");
  rcode_override = ns_r_refused;
  do_both ("mail", "mail.corp.example.com");
  /* Likewise, but with ndots handling.  */
  rcode_override = ns_r_servfail;
  do_both ("file.corp", "file.corp");
  rcode_override = ns_r_refused;
  do_both ("file.corp", "file.corp");

  resolv_test_end (obj);
}

static int
do_test (void)
{
  for (int unconnectable_server = 0; unconnectable_server < 2;
       ++unconnectable_server)
    do_test_all (unconnectable_server);
  return 0;
}

#include <support/test-driver.c>
