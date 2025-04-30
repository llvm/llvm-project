/* Test getnetbyname and getnetbyaddr.
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

#include <netdb.h>
#include <stdlib.h>
#include <string.h>
#include <support/check.h>
#include <support/check_nss.h>
#include <support/resolv_test.h>
#include <support/support.h>
#include <support/xmemstream.h>

static void
send_ptr (struct resolv_response_builder *b,
          const char *qname, uint16_t qclass, uint16_t qtype,
          const char *alias)
{
  resolv_response_init (b, (struct resolv_response_flags) {});
  resolv_response_add_question (b, qname, qclass, qtype);
  resolv_response_section (b, ns_s_an);
  resolv_response_open_record (b, qname, qclass, T_PTR, 0);
  resolv_response_add_name (b, alias);
  resolv_response_close_record (b);
}

static void
handle_code (const struct resolv_response_context *ctx,
             struct resolv_response_builder *b,
             const char *qname, uint16_t qclass, uint16_t qtype,
             int code)
{
  switch (code)
    {
    case 1:
      send_ptr (b, qname, qclass, qtype, "1.in-addr.arpa");
      break;
    case 2:
      send_ptr (b, qname, qclass, qtype, "2.1.in-addr.arpa");
      break;
    case 3:
      send_ptr (b, qname, qclass, qtype, "3.2.1.in-addr.arpa");
      break;
    case 4:
      send_ptr (b, qname, qclass, qtype, "4.3.2.1.in-addr.arpa");
      break;
    case 5:
      /* Test multiple PTR records.  */
      resolv_response_init (b, (struct resolv_response_flags) {});
      resolv_response_add_question (b, qname, qclass, qtype);
      resolv_response_section (b, ns_s_an);
      resolv_response_open_record (b, qname, qclass, T_PTR, 0);
      resolv_response_add_name (b, "127.in-addr.arpa");
      resolv_response_close_record (b);
      resolv_response_open_record (b, qname, qclass, T_PTR, 0);
      resolv_response_add_name (b, "0.in-addr.arpa");
      resolv_response_close_record (b);
      break;
    case 6:
      /* Test skipping of RRSIG record.  */
      resolv_response_init (b, (struct resolv_response_flags) { });
      resolv_response_add_question (b, qname, qclass, qtype);
      resolv_response_section (b, ns_s_an);

      resolv_response_open_record (b, qname, qclass, T_PTR, 0);
      resolv_response_add_name (b, "127.in-addr.arpa");
      resolv_response_close_record (b);

      resolv_response_open_record (b, qname, qclass, 46 /* RRSIG */, 0);
      {
        char buf[500];
        memset (buf, 0x3f, sizeof (buf));
        resolv_response_add_data (b, buf, sizeof (buf));
      }
      resolv_response_close_record (b);

      resolv_response_open_record (b, qname, qclass, T_PTR, 0);
      resolv_response_add_name (b, "0.in-addr.arpa");
      resolv_response_close_record (b);
      break;
    case 7:
      /* Test CNAME handling.  */
      resolv_response_init (b, (struct resolv_response_flags) { });
      resolv_response_add_question (b, qname, qclass, qtype);
      resolv_response_section (b, ns_s_an);
      resolv_response_open_record (b, qname, qclass, T_CNAME, 0);
      resolv_response_add_name (b, "cname.example");
      resolv_response_close_record (b);
      resolv_response_open_record (b, "cname.example", qclass, T_PTR, 0);
      resolv_response_add_name (b, "4.3.2.1.in-addr.arpa");
      resolv_response_close_record (b);
      break;

    case 100:
      resolv_response_init (b, (struct resolv_response_flags) { .rcode = 0, });
      resolv_response_add_question (b, qname, qclass, qtype);
      break;
    case 101:
      resolv_response_init (b, (struct resolv_response_flags)
                            { .rcode = NXDOMAIN, });
      resolv_response_add_question (b, qname, qclass, qtype);
      break;
    case 102:
      resolv_response_init (b, (struct resolv_response_flags) {.rcode = SERVFAIL});
      resolv_response_add_question (b, qname, qclass, qtype);
      break;
    case 103:
      /* Check response length matching.  */
      if (!ctx->tcp)
        {
          resolv_response_init (b, (struct resolv_response_flags) {.tc = true});
          resolv_response_add_question (b, qname, qclass, qtype);
        }
      else
        {
          resolv_response_init (b, (struct resolv_response_flags) {.ancount = 1});
          resolv_response_add_question (b, qname, qclass, qtype);
          resolv_response_section (b, ns_s_an);
          resolv_response_open_record (b, qname, qclass, T_PTR, 0);
          resolv_response_add_name (b, "127.in-addr.arpa");
          resolv_response_close_record (b);
          resolv_response_open_record (b, qname, qclass, T_PTR, 0);
          resolv_response_add_name (b, "example");
          resolv_response_close_record (b);

          resolv_response_open_record (b, qname, qclass, T_PTR, 0);
          size_t to_fill = 65535 - resolv_response_length (b)
            - 2 /* length, "n" */ - 2 /* compression reference */
            - 2 /* RR type */;
          for (size_t i = 0; i < to_fill; ++i)
            resolv_response_add_data (b, "", 1);
          resolv_response_close_record (b);
          resolv_response_add_name (b, "n.example");
          uint16_t rrtype = htons (T_PTR);
          resolv_response_add_data (b, &rrtype, sizeof (rrtype));
        }
      break;
    case 104:
      send_ptr (b, qname, qclass, qtype, "host.example");
      break;
    default:
      FAIL_EXIT1 ("invalid QNAME: %s (code %d)", qname, code);
    }
}

static void
response (const struct resolv_response_context *ctx,
          struct resolv_response_builder *b,
          const char *qname, uint16_t qclass, uint16_t qtype)
{
  int code;
  if (strstr (qname, "in-addr.arpa") == NULL)
    {
      char *tail;
      if (sscanf (qname, "code%d.%ms", &code, &tail) != 2
          || strcmp (tail, "example") != 0)
        FAIL_EXIT1 ("invalid QNAME: %s", qname);
      free (tail);
      handle_code (ctx, b, qname, qclass, qtype, code);
    }
  else
    {
      /* Reverse lookup.  */
      int components[4];
      char *tail;
      if (sscanf (qname, "%d.%d.%d.%d.%ms",
                  components, components + 1, components + 2, components + 3,
                  &tail) != 5
          || strcmp (tail, "in-addr.arpa") != 0)
        FAIL_EXIT1 ("invalid QNAME: %s", qname);
      free (tail);
      handle_code (ctx, b, qname, qclass, qtype, components[3]);
    }
}

static void
check_reverse (int code, const char *expected)
{
  char *query = xasprintf ("code=%d", code);
  check_netent (query, getnetbyaddr (code, AF_INET), expected);
  free (query);
}

/* Test for CVE-2016-3075.  */
static void
check_long_name (void)
{
  struct xmemstream mem;
  xopen_memstream (&mem);

  char label[65];
  memset (label, 'x', 63);
  label[63] = '.';
  label[64] = '\0';
  for (unsigned i = 0; i < 64 * 1024 * 1024 / strlen (label); ++i)
    fprintf (mem.out, "%s", label);

  xfclose_memstream (&mem);

  check_netent ("long name", getnetbyname (mem.buffer),
                "error: NO_RECOVERY\n");

  free (mem.buffer);
}

static int
do_test (void)
{
  struct resolv_test *obj = resolv_test_start
    ((struct resolv_redirect_config)
     {
       .response_callback = response
     });

  /* Lookup by name, success cases.  */
  check_netent ("code1.example", getnetbyname ("code1.example"),
                "alias: 1.in-addr.arpa\n"
                "net: 0x00000001\n");
  check_netent ("code2.example", getnetbyname ("code2.example"),
                "alias: 2.1.in-addr.arpa\n"
                "net: 0x00000102\n");
  check_netent ("code3.example", getnetbyname ("code3.example"),
                "alias: 3.2.1.in-addr.arpa\n"
                "net: 0x00010203\n");
  check_netent ("code4.example", getnetbyname ("code4.example"),
                "alias: 4.3.2.1.in-addr.arpa\n"
                "net: 0x01020304\n");
  check_netent ("code5.example", getnetbyname ("code5.example"),
                "alias: 127.in-addr.arpa\n"
                "alias: 0.in-addr.arpa\n"
                "net: 0x0000007f\n");
  check_netent ("code6.example", getnetbyname ("code6.example"),
                "alias: 127.in-addr.arpa\n"
                "alias: 0.in-addr.arpa\n"
                "net: 0x0000007f\n");
  check_netent ("code7.example", getnetbyname ("code7.example"),
                "alias: 4.3.2.1.in-addr.arpa\n"
                "net: 0x01020304\n");

  /* Lookup by name, failure cases.  */
  check_netent ("code100.example", getnetbyname ("code100.example"),
                "error: NO_ADDRESS\n");
  check_netent ("code101.example", getnetbyname ("code101.example"),
                "error: HOST_NOT_FOUND\n");
  check_netent ("code102.example", getnetbyname ("code102.example"),
                "error: TRY_AGAIN\n");
  check_netent ("code103.example", getnetbyname ("code103.example"),
                "error: NO_RECOVERY\n");
  /* Test bug #17630.  */
  check_netent ("code104.example", getnetbyname ("code104.example"),
                "error: TRY_AGAIN\n");

  /* Lookup by address, success cases.  */
  check_reverse (1,
                 "name: 1.in-addr.arpa\n"
                 "net: 0x00000001\n");
  check_reverse (2,
                 "name: 2.1.in-addr.arpa\n"
                 "net: 0x00000002\n");
  check_reverse (3,
                 "name: 3.2.1.in-addr.arpa\n"
                 "net: 0x00000003\n");
  check_reverse (4,
                 "name: 4.3.2.1.in-addr.arpa\n"
                 "net: 0x00000004\n");
  check_reverse (5,
                 "name: 127.in-addr.arpa\n"
                 "alias: 0.in-addr.arpa\n"
                 "net: 0x00000005\n");
  check_reverse (6,
                 "name: 127.in-addr.arpa\n"
                 "alias: 0.in-addr.arpa\n"
                 "net: 0x00000006\n");
  check_reverse (7,
                 "name: 4.3.2.1.in-addr.arpa\n"
                 "net: 0x00000007\n");

  /* Lookup by address, failure cases.  */
  check_reverse (100,
                 "error: NO_ADDRESS\n");
  check_reverse (101,
                 "error: HOST_NOT_FOUND\n");
  check_reverse (102,
                 "error: TRY_AGAIN\n");
  check_reverse (103,
                 "error: NO_RECOVERY\n");

  check_long_name ();

  resolv_test_end (obj);

  return 0;
}

#include <support/test-driver.c>
