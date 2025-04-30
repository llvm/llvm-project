/* Test name resolution behavior with trailing characters.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

#include <array_length.h>
#include <netdb.h>
#include <support/check.h>
#include <support/check_nss.h>
#include <support/resolv_test.h>
#include <support/support.h>

static void
response (const struct resolv_response_context *ctx,
          struct resolv_response_builder *b,
          const char *qname, uint16_t qclass, uint16_t qtype)
{
  /* The tests are not supposed send any DNS queries.  */
  FAIL_EXIT1 ("unexpected DNS query for %s/%d/%d", qname, qclass, qtype);
}

static int
do_test (void)
{
  struct resolv_test *aux = resolv_test_start
    ((struct resolv_redirect_config)
     {
       .response_callback = response,
     });

  static const char *const queries[] =
    {
     "192.0.2.1 ",
     "192.0.2.2\t",
     "192.0.2.3\n",
     "192.0.2.4 X",
     "192.0.2.5\tY",
     "192.0.2.6\nZ",
     "192.0.2. ",
     "192.0.2.\t",
     "192.0.2.\n",
     "192.0.2. X",
     "192.0.2.\tY",
     "192.0.2.\nZ",
     "2001:db8::1 ",
     "2001:db8::2\t",
     "2001:db8::3\n",
     "2001:db8::4 X",
     "2001:db8::5\tY",
     "2001:db8::6\nZ",
    };
  for (size_t query_idx = 0; query_idx < array_length (queries); ++query_idx)
    {
      const char *query = queries[query_idx];
      struct hostent storage;
      char buf[4096];
      struct hostent *e;

      h_errno = 0;
      TEST_VERIFY (gethostbyname (query) == NULL);
      TEST_COMPARE (h_errno, HOST_NOT_FOUND);

      h_errno = 0;
      e = NULL;
      TEST_COMPARE (gethostbyname_r (query, &storage, buf, sizeof (buf),
                                     &e, &h_errno), 0);
      TEST_VERIFY (e == NULL);
      TEST_COMPARE (h_errno, HOST_NOT_FOUND);

      h_errno = 0;
      TEST_VERIFY (gethostbyname2 (query, AF_INET) == NULL);
      TEST_COMPARE (h_errno, HOST_NOT_FOUND);

      h_errno = 0;
      e = NULL;
      TEST_COMPARE (gethostbyname2_r (query, AF_INET,
                                      &storage, buf, sizeof (buf),
                                     &e, &h_errno), 0);
      TEST_VERIFY (e == NULL);
      TEST_COMPARE (h_errno, HOST_NOT_FOUND);

      h_errno = 0;
      TEST_VERIFY (gethostbyname2 (query, AF_INET6) == NULL);
      TEST_COMPARE (h_errno, HOST_NOT_FOUND);

      h_errno = 0;
      e = NULL;
      TEST_COMPARE (gethostbyname2_r (query, AF_INET6,
                                      &storage, buf, sizeof (buf),
                                     &e, &h_errno), 0);
      TEST_VERIFY (e == NULL);
      TEST_COMPARE (h_errno, HOST_NOT_FOUND);

      static const int gai_flags[] =
        {
         0,
         AI_ADDRCONFIG,
         AI_NUMERICHOST,
         AI_IDN,
         AI_IDN | AI_NUMERICHOST,
         AI_V4MAPPED,
         AI_V4MAPPED | AI_NUMERICHOST,
        };
      for (size_t gai_flags_idx; gai_flags_idx < array_length (gai_flags);
             ++gai_flags_idx)
        {
          struct addrinfo hints = { .ai_flags = gai_flags[gai_flags_idx], };
          struct addrinfo *ai;
          hints.ai_family = AF_INET;
          TEST_COMPARE (getaddrinfo (query, "80", &hints, &ai), EAI_NONAME);
          hints.ai_family = AF_INET6;
          TEST_COMPARE (getaddrinfo (query, "80", &hints, &ai), EAI_NONAME);
          hints.ai_family = AF_UNSPEC;
          TEST_COMPARE (getaddrinfo (query, "80", &hints, &ai), EAI_NONAME);
        }
    };

  resolv_test_end (aux);

  return 0;
}

#include <support/test-driver.c>
