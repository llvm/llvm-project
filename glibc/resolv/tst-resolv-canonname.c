/* Test _nss_dns_getcanonname_r corner cases.
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

#include <dlfcn.h>
#include <errno.h>
#include <gnu/lib-names.h>
#include <netdb.h>
#include <nss.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <support/check.h>
#include <support/resolv_test.h>
#include <support/support.h>

/* _nss_dns_getcanonname_r is not called during regular operation
   because nss_dns directly provides a canonical name, so we have to
   test it directly.  The function pointer is initialized by do_test
   below.  */
static enum nss_status
(*getcanonname) (const char *name, char *buffer, size_t buflen,
                 char **result, int *errnop, int *h_errnop);

static void
response (const struct resolv_response_context *ctx,
          struct resolv_response_builder *b,
          const char *qname, uint16_t qclass, uint16_t qtype)
{
  int code;
  {
    char *tail;
    if (sscanf (qname, "code%d.%ms", &code, &tail) != 2
        || strcmp (tail, "example") != 0)
      FAIL_EXIT1 ("error: invalid QNAME: %s\n", qname);
    free (tail);
  }

  switch (code)
    {
    case 1:
      resolv_response_init (b, (struct resolv_response_flags) {});
      resolv_response_add_question (b, qname, qclass, qtype);
      resolv_response_section (b, ns_s_an);
      resolv_response_open_record (b, "www.example", qclass, qtype, 0);
      resolv_response_add_data (b, "\xC0\x00\x02\x01", 4);
      resolv_response_close_record (b);
      break;
    case 2:
      resolv_response_init (b, (struct resolv_response_flags) {});
      resolv_response_add_question (b, qname, qclass, qtype);
      resolv_response_section (b, ns_s_an);
      if (qtype == T_AAAA)
        {
          resolv_response_open_record (b, "www.example", qclass, qtype, 0);
          resolv_response_add_data (b, "\xC0\x00\x02\x01", 4);
          resolv_response_close_record (b);
          for (int i = 0; i < 30000; ++i)
            resolv_response_add_data (b, "", 1);
        }
      break;
    case 3:
      resolv_response_init (b, (struct resolv_response_flags) {});
      resolv_response_add_question (b, qname, qclass, qtype);
      resolv_response_section (b, ns_s_an);
      if (qtype == T_AAAA)
        {
          resolv_response_open_record (b, "www.example", qclass, qtype, 0);
          resolv_response_add_data (b, "\xC0\x00\x02\x01", 4);
          resolv_response_close_record (b);
        }
      else
        {
          for (int i = 0; i < 30000; ++i)
            resolv_response_add_data (b, "", 1);
        }
      break;
    case 4:
      resolv_response_init (b, (struct resolv_response_flags) {});
      resolv_response_add_question (b, qname, qclass, qtype);
      resolv_response_section (b, ns_s_an);
      resolv_response_open_record (b, qname, qclass, T_CNAME, 0);
      resolv_response_add_name (b, "www.example");
      resolv_response_close_record (b);
      resolv_response_open_record (b, "www.example", qclass, qtype, 0);
      resolv_response_add_data (b, "\xC0\x00\x02\x01", 4);
      resolv_response_close_record (b);
      break;
    case 5:
      resolv_response_init (b, (struct resolv_response_flags) {});
      resolv_response_add_question (b, qname, qclass, qtype);
      resolv_response_section (b, ns_s_an);
      resolv_response_open_record (b, qname, qclass, T_CNAME, 0);
      resolv_response_add_name (b, "www.example");
      resolv_response_close_record (b);
      resolv_response_open_record (b, qname, qclass, T_CNAME, 0);
      resolv_response_add_name (b, "www1.example");
      resolv_response_close_record (b);
      resolv_response_open_record (b, "www1.example", qclass, qtype, 0);
      resolv_response_add_data (b, "\xC0\x00\x02\x01", 4);
      resolv_response_close_record (b);
      break;
    case 6:
      resolv_response_init (b, (struct resolv_response_flags) {});
      resolv_response_add_question (b, qname, qclass, qtype);
      resolv_response_section (b, ns_s_an);
      resolv_response_open_record (b, qname, qclass, T_CNAME, 0);
      resolv_response_add_name (b, "www.example");
      resolv_response_close_record (b);
      resolv_response_open_record (b, qname, qclass, 46 /* RRSIG */, 0);
      resolv_response_add_name (b, ".");
      resolv_response_close_record (b);
      resolv_response_open_record (b, "www.example", qclass, qtype, 0);
      resolv_response_add_data (b, "\xC0\x00\x02\x01", 4);
      resolv_response_close_record (b);
      break;
    case 102:
      if (!ctx->tcp)
        {
          resolv_response_init (b, (struct resolv_response_flags) {.tc = true});
          resolv_response_add_question (b, qname, qclass, qtype);
        }
      else
        {
          resolv_response_init
            (b, (struct resolv_response_flags) {.ancount = 1});
          resolv_response_add_question (b, qname, qclass, qtype);
          resolv_response_section (b, ns_s_an);
          resolv_response_open_record (b, qname, qclass, T_CNAME, 0);
          size_t to_fill = 65535 - resolv_response_length (b)
            - 2 /* length, "n" */ - 2 /* compression reference */
            - 2 /* RR type */;
          for (size_t i = 0; i < to_fill; ++i)
            resolv_response_add_data (b, "", 1);
          resolv_response_close_record (b);
          resolv_response_add_name (b, "n.example");
          uint16_t rrtype = htons (T_CNAME);
          resolv_response_add_data (b, &rrtype, sizeof (rrtype));
        }
      break;
    case 103:
      /* NODATA repsonse.  */
      resolv_response_init (b, (struct resolv_response_flags) {});
      resolv_response_add_question (b, qname, qclass, qtype);
      break;
    case 104:
      resolv_response_init (b, (struct resolv_response_flags) {.ancount = 1});
      resolv_response_add_question (b, qname, qclass, qtype);
      /* No RR metadata.  */
      resolv_response_add_name (b, "www.example");
      break;
    case 105:
      if (qtype == T_A)
        {
          resolv_response_init (b, (struct resolv_response_flags) {});
          resolv_response_add_question (b, qname, qclass, qtype);
          /* No data, trigger AAAA query.  */
        }
      else
        {
          resolv_response_init
            (b, (struct resolv_response_flags) {.ancount = 1});
          resolv_response_add_question (b, qname, qclass, qtype);
          /* No RR metadata.  */
          resolv_response_add_name
            (b, "long-name-exceed-previously-initialized-buffer.example");
        }
      break;
    case 106:
      resolv_response_init (b, (struct resolv_response_flags) {.ancount = 1});
      resolv_response_add_question (b, qname, qclass, qtype);
      /* No RR metadata.  */
      resolv_response_add_name (b, "www.example");
      resolv_response_add_data (b, "\xff\xff", 2);
      break;
    case 107:
      if (qtype == T_A)
        {
          resolv_response_init (b, (struct resolv_response_flags) {});
          resolv_response_add_question (b, qname, qclass, qtype);
          /* No data, trigger AAAA query.  */
        }
      else
        {
          resolv_response_init
            (b, (struct resolv_response_flags) {.ancount = 1});
          resolv_response_add_question (b, qname, qclass, qtype);
          /* No RR metadata.  */
          resolv_response_add_name (b, "www.example");
          resolv_response_add_data (b, "\xff\xff", 2);
        }
      break;
    default:
      FAIL_EXIT1 ("error: invalid QNAME: %s (code %d)\n", qname, code);
    }
}

static void
check (int code, const char *expected)
{
  char qname[200];
  snprintf (qname, sizeof (qname), "code%d.example", code);
  char *result;
  enum nss_status status;
  {
    enum { buffer_size = 4096 };
    char *buffer = xmalloc (buffer_size);
    char *temp_result;
    int temp_errno;
    int temp_herrno;
    status = getcanonname
      (qname, buffer, buffer_size, &temp_result, &temp_errno, &temp_herrno);
    if (status == NSS_STATUS_SUCCESS)
      result = xstrdup (temp_result);
    else
      {
        errno = temp_errno;
        h_errno = temp_herrno;
      }
    free (buffer);
  }

  if (status == NSS_STATUS_SUCCESS)
    {
      if (expected != NULL)
        {
          if (strcmp (result, expected) != 0)
            {
              support_record_failure ();
              printf ("error: getcanonname (%s) failed\n", qname);
              printf ("error:  expected: %s\n", expected);
              printf ("error:  actual:   %s\n", result);
              free (result);
              return;
            }
        }
      else
        {
          support_record_failure ();
          printf ("error: getcanonname (%s) unexpected success\n", qname);
          printf ("error:  actual:   %s\n", result);
          free (result);
          return;
        }
      free (result);
    }
  else
    {
      if (expected != NULL)
        {
          support_record_failure ();
          printf ("error: getcanonname (%s) failed\n", qname);
          printf ("error:  expected: %s\n", expected);
          return;
        }
    }
}


static int
do_test (void)
{
  void *nss_dns_handle = dlopen (LIBNSS_DNS_SO, RTLD_LAZY);
  if (nss_dns_handle == NULL)
    FAIL_EXIT1 ("could not dlopen %s: %s", LIBNSS_DNS_SO, dlerror ());
  {
    const char *func = "_nss_dns_getcanonname_r";
    void *ptr = dlsym (nss_dns_handle, func);
    if (ptr == NULL)
      FAIL_EXIT1 ("could not look up %s: %s", func, dlerror ());
    getcanonname = ptr;
  }

  struct resolv_test *aux = resolv_test_start
    ((struct resolv_redirect_config)
     {
       .response_callback = response,
     });

  check (1, "www.example");
  check (2, "www.example");
  check (3, "www.example");
  check (4, "www.example");
  check (5, "www1.example");

  /* This should really result in "www.example", but the fake RRSIG
     record causes the current implementation to stop parsing.  */
  check (6, NULL);

  for (int i = 102; i <= 107; ++i)
  check (i, NULL);

  resolv_test_end (aux);

  TEST_VERIFY (dlclose (nss_dns_handle) == 0);
  return 0;
}

#include <support/test-driver.c>
