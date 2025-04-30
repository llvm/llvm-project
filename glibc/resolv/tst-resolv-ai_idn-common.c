/* Common code for AI_IDN/NI_IDN tests.
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

/* Before including this file, TEST_USE_UTF8 must be defined to 1 or
   0, depending on whether a UTF-8 locale is used or a Latin-1
   locale.  */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <support/check.h>
#include <support/check_nss.h>
#include <support/resolv_test.h>
#include <support/support.h>

/* Name of the shared object for libidn2.  */
#define LIBIDN2_SONAME "libidn2.so.0"

#if TEST_USE_UTF8
/* UTF-8 encoding of "nämchen" (German for “namelet”).  */
# define NAEMCHEN "n\xC3\xA4mchen"

/* UTF-8 encoding of "שם" (Hebrew for “name”).  */
# define SHEM "\xD7\xA9\xD7\x9D"

/* UTF-8 encoding of "buße" (German for “penance”).  This used to be
   encoded as "busse" (“busses”) in IDNA2003.  */
# define BUSSE "bu\xC3\x9F""e"

#else
/* Latin-1 encodings, as far as they are available.  */

# define NAEMCHEN "n\xE4mchen"
# define BUSSE "bu\xDF""e"

#endif

/* IDNA encoding of NAEMCHEN.  */
#define NAEMCHEN_IDNA "xn--nmchen-bua"

/* IDNA encoding of NAEMCHEN "_zwo".  */
#define NAEMCHEN_ZWO_IDNA "xn--nmchen_zwo-q5a"

/* IDNA encoding of SHEM.  */
#define SHEM_IDNA "xn--iebx"

/* IDNA encoding of BUSSE.  */
#define BUSSE_IDNA "xn--bue-6ka"

/* IDNA encoding of "שם1".  */
#define SHEM1_IDNA "xn--1-qic9a"

/* Another IDNA name.  */
#define ANDERES_NAEMCHEN "anderes-" NAEMCHEN
#define ANDERES_NAEMCHEN_IDNA "xn--anderes-nmchen-eib"

/* Controls the kind of test data in a PTR lookup response.  */
enum gni_test
  {
    gni_non_idn_name,
    gni_non_idn_cname_to_non_idn_name,
    gni_non_idn_cname_to_idn_name,
    gni_idn_name,
    gni_idn_shem,
    gni_idn_shem1,
    gni_idn_cname_to_non_idn_name,
    gni_idn_cname_to_idn_name,
    gni_invalid_idn_1,
    gni_invalid_idn_2,
  };

/* Called from response below.  The LSB (first byte) controls what
   goes into the response, see enum gni_test.  */
static void
response_ptr (const struct resolv_response_context *ctx,
              struct resolv_response_builder *b, const char *qname)
{
  int comp[4] = { 0 };
  TEST_COMPARE (sscanf (qname, "%d.%d.%d.%d.in-addr.arpa",
                        &comp[0], &comp[1], &comp[2], &comp[3]), 4);
  const char *next_name;
  switch ((enum gni_test) comp[0])
    {
    /* First name in response is non-IDN name.  */
    case gni_non_idn_name:
      resolv_response_open_record (b, qname, C_IN, T_PTR, 0);
      resolv_response_add_name (b, "non-idn.example");
      resolv_response_close_record (b);
      return;
    case gni_non_idn_cname_to_non_idn_name:
      resolv_response_open_record (b, qname, C_IN, T_CNAME, 0);
      next_name = "non-idn-cname.example";
      resolv_response_add_name (b, next_name);
      resolv_response_close_record (b);
      resolv_response_open_record (b, next_name, C_IN, T_PTR, 0);
      resolv_response_add_name (b, "non-idn-name.example");
      resolv_response_close_record (b);
      return;
    case gni_non_idn_cname_to_idn_name:
      resolv_response_open_record (b, qname, C_IN, T_CNAME, 0);
      next_name = "non-idn-cname.example";
      resolv_response_add_name (b, next_name);
      resolv_response_close_record (b);
      resolv_response_open_record (b, next_name, C_IN, T_PTR, 0);
      resolv_response_add_name (b, NAEMCHEN_IDNA ".example");
      resolv_response_close_record (b);
      return;

    /* First name in response is IDN name.  */
    case gni_idn_name:
      resolv_response_open_record (b, qname, C_IN, T_PTR, 0);
      resolv_response_add_name (b, "xn--nmchen-bua.example");
      resolv_response_close_record (b);
      return;
    case gni_idn_shem:
      resolv_response_open_record (b, qname, C_IN, T_PTR, 0);
      resolv_response_add_name (b, SHEM_IDNA ".example");
      resolv_response_close_record (b);
      return;
    case gni_idn_shem1:
      resolv_response_open_record (b, qname, C_IN, T_PTR, 0);
      resolv_response_add_name (b, SHEM1_IDNA ".example");
      resolv_response_close_record (b);
      return;
    case gni_idn_cname_to_non_idn_name:
      resolv_response_open_record (b, qname, C_IN, T_CNAME, 0);
      next_name = NAEMCHEN_IDNA ".example";
      resolv_response_add_name (b, next_name);
      resolv_response_close_record (b);
      resolv_response_open_record (b, next_name, C_IN, T_PTR, 0);
      resolv_response_add_name (b, "non-idn-name.example");
      resolv_response_close_record (b);
      return;
    case gni_idn_cname_to_idn_name:
      resolv_response_open_record (b, qname, C_IN, T_CNAME, 0);
      next_name = NAEMCHEN_IDNA ".example";
      resolv_response_add_name (b, next_name);
      resolv_response_close_record (b);
      resolv_response_open_record (b, next_name, C_IN, T_PTR, 0);
      resolv_response_add_name (b, ANDERES_NAEMCHEN_IDNA ".example");
      resolv_response_close_record (b);
      return;

    /* Invalid IDN encodings.  */
    case gni_invalid_idn_1:
      resolv_response_open_record (b, qname, C_IN, T_PTR, 0);
      resolv_response_add_name (b, "xn---.example");
      resolv_response_close_record (b);
      return;
    case gni_invalid_idn_2:
      resolv_response_open_record (b, qname, C_IN, T_PTR, 0);
      resolv_response_add_name (b, "xn--x.example");
      resolv_response_close_record (b);
      return;
    }
  FAIL_EXIT1 ("invalid PTR query: %s", qname);
}

/* For PTR responses, see above.  A/AAAA queries can request
   additional CNAMEs in the response by include ".cname." and
   ".idn-cname." in the query.  The LSB in the address contains the
   first byte of the QNAME.  */
static void
response (const struct resolv_response_context *ctx,
          struct resolv_response_builder *b,
          const char *qname, uint16_t qclass, uint16_t qtype)
{
  TEST_VERIFY_EXIT (qclass == C_IN);

  for (const char *p = qname; *p != '\0'; ++p)
    if (!(('0' <= *p && *p <= '9')
          || ('a' <= *p && *p <= 'z')
          || ('A' <= *p && *p <= 'Z')
          || *p == '.' || *p == '-' || *p == '_'))
      {
        /* Non-ASCII query.  Reply with NXDOMAIN.  */
        struct resolv_response_flags flags = { .rcode = 3 };
        resolv_response_init (b, flags);
        resolv_response_add_question (b, qname, qclass, qtype);
        return;
      }

  struct resolv_response_flags flags = { 0 };
  resolv_response_init (b, flags);
  resolv_response_add_question (b, qname, qclass, qtype);
  resolv_response_section (b, ns_s_an);

  if (qtype == T_PTR)
    {
      response_ptr (ctx, b, qname);
      return;
    }

  bool with_cname = strstr (qname, ".cname.") != NULL;
  bool with_idn_cname = strstr (qname, ".idn-cname.") != NULL;

  const char *next_name = qname;
  if (with_cname)
    {
      next_name = "non-idn-cname.example";
      resolv_response_open_record (b, qname, C_IN, T_CNAME, 0);
      resolv_response_add_name (b, next_name);
      resolv_response_close_record (b);
    }
  if (with_idn_cname)
    {
      const char *previous_name = next_name;
      next_name = ANDERES_NAEMCHEN_IDNA ".example";
      resolv_response_open_record (b, previous_name, C_IN, T_CNAME, 0);
      resolv_response_add_name (b, next_name);
      resolv_response_close_record (b);
    }

  resolv_response_open_record (b, next_name, C_IN, qtype, 0);
  switch (qtype)
    {
    case T_A:
      {
        char addr[4] = { 192, 0, 2, qname[0] };
        resolv_response_add_data (b, &addr, sizeof (addr));
      }
      break;
    case T_AAAA:
      {
        char addr[16]
          = { 0x20, 0x01, 0xd, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              qname[0] };
        resolv_response_add_data (b, &addr, sizeof (addr));
      }
      break;
    default:
      FAIL_EXIT1 ("invalid qtype: %d", qtype);
    }
  resolv_response_close_record (b);
}

/* Check the result of a getaddrinfo call.  */
static void
check_ai (const char *name, int ai_flags, const char *expected)
{
  struct addrinfo hints =
    {
      .ai_flags = ai_flags,
      .ai_family = AF_INET,
      .ai_socktype = SOCK_STREAM,
    };
  struct addrinfo *ai;
  char *query = xasprintf ("%s:80 AF_INET/0x%x", name, ai_flags);
  int ret = getaddrinfo (name, "80", &hints, &ai);
  check_addrinfo (query, ai, ret, expected);
  if (ret == 0)
    freeaddrinfo (ai);
  free (query);
}

/* Run one getnameinfo test.  FLAGS is automatically augmented with
   NI_NUMERICSERV.  */
static void
gni_test (enum gni_test code, unsigned int flags, const char *expected)
{
  struct sockaddr_in sin =
    {
      .sin_family = AF_INET,
      .sin_port = htons (80),
      .sin_addr = { htonl (0xc0000200 | code) }, /* 192.0.2.0/24 network.  */
    };
  char host[1024];
  char service[1024];
  int ret = getnameinfo ((const struct sockaddr *) &sin, sizeof (sin),
                         host, sizeof (host), service, sizeof (service),
                         flags| NI_NUMERICSERV);
  if (ret != 0)
    {
      if (expected == NULL)
        TEST_COMPARE (ret, EAI_IDN_ENCODE);
      else
        {
          support_record_failure ();
          printf ("error: getnameinfo failed (code %d, flags 0x%x): %s (%d)\n",
                  (int) code, flags, gai_strerror (ret), ret);
        }
    }
  else if (ret == 0 && expected == NULL)
    {
      support_record_failure ();
      printf ("error: getnameinfo unexpected success (code %d, flags 0x%x)\n",
              (int) code, flags);
    }
  else if (strcmp (host, expected) != 0 || strcmp (service, "80") != 0)
    {
      support_record_failure ();
      printf ("error: getnameinfo test failure (code %d, flags 0x%x)\n"
              "  expected host:    \"%s\"\n"
              "  expected service: \"80\"\n"
              "  actual host:      \"%s\"\n"
              "  actual service:   \"%s\"\n",
              (int) code, flags, expected, host, service);
    }
}

/* Tests for getaddrinfo which assume a working libidn2 library.  */
__attribute__ ((unused))
static void
gai_tests_with_libidn2 (void)
{
  /* No CNAME.  */
  check_ai ("non-idn.example", 0,
            "address: STREAM/TCP 192.0.2.110 80\n");
  check_ai ("non-idn.example", AI_IDN,
            "flags: AI_IDN\n"
            "address: STREAM/TCP 192.0.2.110 80\n");
  check_ai ("non-idn.example", AI_IDN | AI_CANONNAME | AI_CANONIDN,
            "flags: AI_CANONNAME AI_IDN AI_CANONIDN\n"
            "canonname: non-idn.example\n"
            "address: STREAM/TCP 192.0.2.110 80\n");

  check_ai (NAEMCHEN ".example", 0,
            "error: Name or service not known\n");
  check_ai (NAEMCHEN ".example", AI_IDN,
            "flags: AI_IDN\n"
            "address: STREAM/TCP 192.0.2.120 80\n");
  check_ai (NAEMCHEN ".example", AI_IDN | AI_CANONNAME | AI_CANONIDN,
            "flags: AI_CANONNAME AI_IDN AI_CANONIDN\n"
            "canonname: " NAEMCHEN ".example\n"
            "address: STREAM/TCP 192.0.2.120 80\n");

#if TEST_USE_UTF8
  check_ai (SHEM ".example", 0,
            "error: Name or service not known\n");
  check_ai (SHEM ".example", AI_IDN,
            "flags: AI_IDN\n"
            "address: STREAM/TCP 192.0.2.120 80\n");
  check_ai (SHEM ".example", AI_IDN | AI_CANONNAME | AI_CANONIDN,
            "flags: AI_CANONNAME AI_IDN AI_CANONIDN\n"
            "canonname: " SHEM ".example\n"
            "address: STREAM/TCP 192.0.2.120 80\n");
  check_ai (SHEM ".example", AI_IDN | AI_CANONNAME,
            "flags: AI_CANONNAME AI_IDN\n"
            "canonname: " SHEM_IDNA ".example\n"
            "address: STREAM/TCP 192.0.2.120 80\n");
  check_ai (SHEM "1.example", AI_IDN,
            "flags: AI_IDN\n"
            "address: STREAM/TCP 192.0.2.120 80\n");
  check_ai (SHEM "1.example", AI_IDN | AI_CANONNAME | AI_CANONIDN,
            "flags: AI_CANONNAME AI_IDN AI_CANONIDN\n"
            "canonname: " SHEM "1.example\n"
            "address: STREAM/TCP 192.0.2.120 80\n");
  check_ai (SHEM "1.example", AI_IDN | AI_CANONNAME,
            "flags: AI_CANONNAME AI_IDN\n"
            "canonname: " SHEM1_IDNA ".example\n"
            "address: STREAM/TCP 192.0.2.120 80\n");
#endif

  /* Check that non-transitional mode is active.  German sharp S
     should not turn into SS.  */
  check_ai (BUSSE ".example", 0,
            "error: Name or service not known\n");
  check_ai (BUSSE ".example", AI_IDN,
            "flags: AI_IDN\n"
            "address: STREAM/TCP 192.0.2.120 80\n");
  check_ai (BUSSE ".example", AI_IDN | AI_CANONNAME,
            "flags: AI_CANONNAME AI_IDN\n"
            "canonname: " BUSSE_IDNA ".example\n"
            "address: STREAM/TCP 192.0.2.120 80\n");
  check_ai (BUSSE ".example", AI_IDN | AI_CANONNAME | AI_CANONIDN,
            "flags: AI_CANONNAME AI_IDN AI_CANONIDN\n"
            "canonname: " BUSSE ".example\n"
            "address: STREAM/TCP 192.0.2.120 80\n");

  /* Check that Unicode TR 46 mode is active.  Underscores should be
     permitted in IDNA components.  */
  check_ai (NAEMCHEN "_zwo.example", 0,
            "error: Name or service not known\n");
  check_ai (NAEMCHEN "_zwo.example", AI_IDN,
            "flags: AI_IDN\n"
            "address: STREAM/TCP 192.0.2.120 80\n");
  check_ai (NAEMCHEN "_zwo.example", AI_IDN | AI_CANONNAME,
            "flags: AI_CANONNAME AI_IDN\n"
            "canonname: " NAEMCHEN_ZWO_IDNA ".example\n"
            "address: STREAM/TCP 192.0.2.120 80\n");
  check_ai (NAEMCHEN "_zwo.example", AI_IDN | AI_CANONNAME | AI_CANONIDN,
            "flags: AI_CANONNAME AI_IDN AI_CANONIDN\n"
            "canonname: " NAEMCHEN "_zwo.example\n"
            "address: STREAM/TCP 192.0.2.120 80\n");

  /* No CNAME, but already IDN-encoded.  */
  check_ai (NAEMCHEN_IDNA ".example", 0,
            "address: STREAM/TCP 192.0.2.120 80\n");
  check_ai (NAEMCHEN_IDNA ".example", AI_IDN,
            "flags: AI_IDN\n"
            "address: STREAM/TCP 192.0.2.120 80\n");
  check_ai (NAEMCHEN_IDNA ".example", AI_IDN | AI_CANONNAME,
            "flags: AI_CANONNAME AI_IDN\n"
            "canonname: " NAEMCHEN_IDNA ".example\n"
            "address: STREAM/TCP 192.0.2.120 80\n");
  check_ai (NAEMCHEN_IDNA ".example", AI_IDN | AI_CANONNAME | AI_CANONIDN,
            "flags: AI_CANONNAME AI_IDN AI_CANONIDN\n"
            "canonname: " NAEMCHEN ".example\n"
            "address: STREAM/TCP 192.0.2.120 80\n");
  check_ai (SHEM_IDNA ".example", 0,
            "address: STREAM/TCP 192.0.2.120 80\n");
  check_ai (SHEM_IDNA ".example", AI_IDN,
            "flags: AI_IDN\n"
            "address: STREAM/TCP 192.0.2.120 80\n");
  check_ai (SHEM_IDNA ".example", AI_IDN | AI_CANONNAME,
            "flags: AI_CANONNAME AI_IDN\n"
            "canonname: " SHEM_IDNA ".example\n"
            "address: STREAM/TCP 192.0.2.120 80\n");
#if TEST_USE_UTF8
  check_ai (SHEM_IDNA ".example", AI_IDN | AI_CANONNAME | AI_CANONIDN,
            "flags: AI_CANONNAME AI_IDN AI_CANONIDN\n"
            "canonname: " SHEM ".example\n"
            "address: STREAM/TCP 192.0.2.120 80\n");
#else
  check_ai (SHEM_IDNA ".example", AI_IDN | AI_CANONNAME | AI_CANONIDN,
            "flags: AI_CANONNAME AI_IDN AI_CANONIDN\n"
            "canonname: " SHEM_IDNA ".example\n"
            "address: STREAM/TCP 192.0.2.120 80\n");
#endif

  /* Invalid IDNA canonical name is returned as-is.  */
  check_ai ("xn---.example", AI_CANONNAME | AI_CANONIDN,
            "flags: AI_CANONNAME AI_CANONIDN\n"
            "canonname: xn---.example\n"
            "address: STREAM/TCP 192.0.2.120 80\n");

  /* Non-IDN CNAME.  */
  check_ai ("with.cname.example", 0,
            "address: STREAM/TCP 192.0.2.119 80\n");
  check_ai ("with.cname.example", AI_IDN,
            "flags: AI_IDN\n"
            "address: STREAM/TCP 192.0.2.119 80\n");
  check_ai ("with.cname.example", AI_IDN | AI_CANONNAME | AI_CANONIDN,
            "flags: AI_CANONNAME AI_IDN AI_CANONIDN\n"
            "canonname: non-idn-cname.example\n"
            "address: STREAM/TCP 192.0.2.119 80\n");

  check_ai ("with.cname." NAEMCHEN ".example", 0,
            "error: Name or service not known\n");
  check_ai ("with.cname." NAEMCHEN ".example", AI_IDN,
            "flags: AI_IDN\n"
            "address: STREAM/TCP 192.0.2.119 80\n");
  check_ai ("with.cname." NAEMCHEN ".example", AI_IDN | AI_CANONNAME,
            "flags: AI_CANONNAME AI_IDN\n"
            "canonname: non-idn-cname.example\n"
            "address: STREAM/TCP 192.0.2.119 80\n");
  check_ai ("with.cname." NAEMCHEN ".example",
            AI_IDN | AI_CANONNAME | AI_CANONIDN,
            "flags: AI_CANONNAME AI_IDN AI_CANONIDN\n"
            "canonname: non-idn-cname.example\n"
            "address: STREAM/TCP 192.0.2.119 80\n");

  /* IDN CNAME.  */
  check_ai ("With.idn-cname.example", 0,
            "address: STREAM/TCP 192.0.2.87 80\n");
  check_ai ("With.idn-cname.example", AI_IDN,
            "flags: AI_IDN\n"
            "address: STREAM/TCP 192.0.2.87 80\n");
  check_ai ("With.idn-cname.example", AI_IDN | AI_CANONNAME,
            "flags: AI_CANONNAME AI_IDN\n"
            "canonname: " ANDERES_NAEMCHEN_IDNA ".example\n"
            "address: STREAM/TCP 192.0.2.87 80\n");
  check_ai ("With.idn-cname.example",
            AI_IDN | AI_CANONNAME | AI_CANONIDN,
            "flags: AI_CANONNAME AI_IDN AI_CANONIDN\n"
            "canonname: " ANDERES_NAEMCHEN ".example\n"
            "address: STREAM/TCP 192.0.2.87 80\n");

  check_ai ("With.idn-cname." NAEMCHEN ".example", 0,
            "error: Name or service not known\n");
  check_ai ("With.idn-cname." NAEMCHEN ".example", AI_IDN,
            "flags: AI_IDN\n"
            "address: STREAM/TCP 192.0.2.119 80\n");
  check_ai ("With.idn-cname." NAEMCHEN ".example", AI_IDN | AI_CANONNAME,
            "flags: AI_CANONNAME AI_IDN\n"
            "canonname: " ANDERES_NAEMCHEN_IDNA ".example\n"
            "address: STREAM/TCP 192.0.2.119 80\n");
  check_ai ("With.idn-cname." NAEMCHEN ".example",
            AI_IDN | AI_CANONNAME | AI_CANONIDN,
            "flags: AI_CANONNAME AI_IDN AI_CANONIDN\n"
            "canonname: " ANDERES_NAEMCHEN ".example\n"
            "address: STREAM/TCP 192.0.2.119 80\n");

  /* Non-IDN to IDN CNAME chain.  */
  check_ai ("both.cname.idn-cname.example", 0,
            "address: STREAM/TCP 192.0.2.98 80\n");
  check_ai ("both.cname.idn-cname.example", AI_IDN,
            "flags: AI_IDN\n"
            "address: STREAM/TCP 192.0.2.98 80\n");
  check_ai ("both.cname.idn-cname.example", AI_IDN | AI_CANONNAME,
            "flags: AI_CANONNAME AI_IDN\n"
            "canonname: " ANDERES_NAEMCHEN_IDNA ".example\n"
            "address: STREAM/TCP 192.0.2.98 80\n");
  check_ai ("both.cname.idn-cname.example",
            AI_IDN | AI_CANONNAME | AI_CANONIDN,
            "flags: AI_CANONNAME AI_IDN AI_CANONIDN\n"
            "canonname: " ANDERES_NAEMCHEN ".example\n"
            "address: STREAM/TCP 192.0.2.98 80\n");

  check_ai ("both.cname.idn-cname." NAEMCHEN ".example", 0,
            "error: Name or service not known\n");
  check_ai ("both.cname.idn-cname." NAEMCHEN ".example", AI_IDN,
            "flags: AI_IDN\n"
            "address: STREAM/TCP 192.0.2.98 80\n");
  check_ai ("both.cname.idn-cname." NAEMCHEN ".example",
            AI_IDN | AI_CANONNAME,
            "flags: AI_CANONNAME AI_IDN\n"
            "canonname: " ANDERES_NAEMCHEN_IDNA ".example\n"
            "address: STREAM/TCP 192.0.2.98 80\n");
  check_ai ("both.cname.idn-cname." NAEMCHEN ".example",
            AI_IDN | AI_CANONNAME | AI_CANONIDN,
            "flags: AI_CANONNAME AI_IDN AI_CANONIDN\n"
            "canonname: " ANDERES_NAEMCHEN ".example\n"
            "address: STREAM/TCP 192.0.2.98 80\n");
}

/* Tests for getnameinfo which assume a working libidn2 library.  */
__attribute__ ((unused))
static void
gni_tests_with_libidn2 (void)
{
  gni_test (gni_non_idn_name, 0, "non-idn.example");
  gni_test (gni_non_idn_name, NI_IDN, "non-idn.example");
  gni_test (gni_non_idn_name, NI_NUMERICHOST, "192.0.2.0");
  gni_test (gni_non_idn_name, NI_NUMERICHOST | NI_IDN, "192.0.2.0");

  gni_test (gni_non_idn_cname_to_non_idn_name, 0, "non-idn-name.example");
  gni_test (gni_non_idn_cname_to_non_idn_name, NI_IDN, "non-idn-name.example");

  gni_test (gni_non_idn_cname_to_idn_name, 0, NAEMCHEN_IDNA ".example");
  gni_test (gni_non_idn_cname_to_idn_name, NI_IDN, NAEMCHEN ".example");

  gni_test (gni_idn_name, 0, NAEMCHEN_IDNA ".example");
  gni_test (gni_idn_name, NI_IDN, NAEMCHEN ".example");
  gni_test (gni_idn_shem, 0, SHEM_IDNA ".example");
  gni_test (gni_idn_shem1, 0, SHEM1_IDNA ".example");
#if TEST_USE_UTF8
  gni_test (gni_idn_shem, NI_IDN, SHEM ".example");
  gni_test (gni_idn_shem1, NI_IDN, SHEM "1.example");
#else
  gni_test (gni_idn_shem, NI_IDN, SHEM_IDNA ".example");
  gni_test (gni_idn_shem1, NI_IDN, SHEM1_IDNA ".example");
#endif

  gni_test (gni_idn_cname_to_non_idn_name, 0, "non-idn-name.example");
  gni_test (gni_idn_cname_to_non_idn_name, NI_IDN, "non-idn-name.example");

  gni_test (gni_idn_cname_to_idn_name, 0, ANDERES_NAEMCHEN_IDNA ".example");
  gni_test (gni_idn_cname_to_idn_name, NI_IDN, ANDERES_NAEMCHEN ".example");

  /* Test encoding errors.  */
  gni_test (gni_invalid_idn_1, 0, "xn---.example");
  gni_test (gni_invalid_idn_1, NI_IDN, "xn---.example");
  gni_test (gni_invalid_idn_2, 0, "xn--x.example");
  gni_test (gni_invalid_idn_2, NI_IDN, "xn--x.example");
}
