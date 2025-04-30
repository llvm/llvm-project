/* Tests for __inet6_scopeid_pton and IPv6 scopes in getaddrinfo.
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

#include <arpa/inet.h>
#include <inttypes.h>
#include <net-internal.h>
#include <net/if.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <support/check.h>
#include <support/support.h>
#include <support/test-driver.h>

/* An interface which is known to the system.  */
static const char *interface_name;
static uint32_t interface_index;

/* Initiale the variables above.  */
static void
setup_interface (void)
{
  struct if_nameindex *list = if_nameindex ();
  if (list != NULL && list[0].if_index != 0 && list[0].if_name[0] != '\0')
    {
      interface_name = list[0].if_name;
      interface_index = list[0].if_index;
    }
}

/* Convert ADDRESS to struct in6_addr.  */
static struct in6_addr
from_string (const char *address)
{
  struct in6_addr addr;
  if (inet_pton (AF_INET6, address, &addr) != 1)
    FAIL_EXIT1 ("inet_pton (\"%s\")", address);
  return addr;
}

/* Invoke getaddrinfo to parse ADDRESS%SCOPE.  Return true if
   getaddrinfo was successful.  */
static bool
call_gai (int family, const char *address, const char *scope,
          struct sockaddr_in6 *result)
{
  struct addrinfo hints =
    {
      .ai_family = family,
      .ai_flags = AI_NUMERICHOST,
      .ai_socktype = SOCK_DGRAM,
      .ai_protocol = IPPROTO_UDP,
    };
  char *fulladdr = xasprintf ("%s%%%s", address, scope);
  struct addrinfo *ai = NULL;
  int ret = getaddrinfo (fulladdr, NULL, &hints, &ai);
  if (ret == EAI_ADDRFAMILY || ret == EAI_NONAME)
    {
      if (test_verbose > 0)
        printf ("info: getaddrinfo (\"%s\"): %s (%d)\n",
                fulladdr, gai_strerror (ret), ret);
      free (fulladdr);
      return false;
    }
  if (ret != 0)
    FAIL_EXIT1 ("getaddrinfo (\"%s\"): %s (%d)\n",
                fulladdr, gai_strerror (ret), ret);
  TEST_VERIFY_EXIT (ai != NULL);
  TEST_VERIFY_EXIT (ai->ai_addrlen == sizeof (*result));
  TEST_VERIFY (ai->ai_family == AF_INET6);
  TEST_VERIFY (ai->ai_next == NULL);
  memcpy (result, ai->ai_addr, sizeof (*result));
  free (fulladdr);
  freeaddrinfo (ai);
  return true;
}

/* Verify that a successful call to getaddrinfo returned the expected
   scope data.  */
static void
check_ai (const char *what, const char *addr_string, const char *scope_string,
          const struct sockaddr_in6 *sa,
          const struct in6_addr *addr, uint32_t scope)
{
  if (memcmp (addr, &sa->sin6_addr, sizeof (*addr)) != 0)
    {
      support_record_failure ();
      printf ("error: getaddrinfo %s address mismatch for %s%%%s\n",
              what, addr_string, scope_string);
    }
  if (sa->sin6_scope_id != scope)
    {
      support_record_failure ();
      printf ("error: getaddrinfo %s scope mismatch for %s%%%s\n"
              "  expected: %" PRIu32 "\n"
              "  actual:   %" PRIu32 "\n",
              what, addr_string, scope_string, scope, sa->sin6_scope_id);
    }
}

/* Check a single address were we expected a failure.  */
static void
expect_failure (const char *address, const char *scope)
{
  if (test_verbose > 0)
    printf ("info: expecting failure for %s%%%s\n", address, scope);
  struct in6_addr addr = from_string (address);
  uint32_t result = 1234;
  if (__inet6_scopeid_pton (&addr, scope, &result) == 0)
    {
      support_record_failure ();
      printf ("error: unexpected success for %s%%%s\n",
              address, scope);
    }
  if (result != 1234)
    {
      support_record_failure ();
      printf ("error: unexpected result update for %s%%%s\n",
              address, scope);
    }

  struct sockaddr_in6 sa;
  if (call_gai (AF_UNSPEC, address, scope, &sa))
    {
      support_record_failure ();
      printf ("error: unexpected getaddrinfo success for %s%%%s (AF_UNSPEC)\n",
              address, scope);
    }
  if (call_gai (AF_INET6, address, scope, &sa))
    {
      support_record_failure ();
      printf ("error: unexpected getaddrinfo success for %s%%%s (AF_INET6)\n",
              address, scope);
    }
}

/* Check a single address were we expected a success.  */
static void
expect_success (const char *address, const char *scope, uint32_t expected)
{
  if (test_verbose > 0)
    printf ("info: expecting success for %s%%%s\n", address, scope);
  struct in6_addr addr = from_string (address);
  uint32_t actual = expected + 1;
  if (__inet6_scopeid_pton (&addr, scope, &actual) != 0)
    {
      support_record_failure ();
      printf ("error: unexpected failure for %s%%%s\n",
              address, scope);
    }
  if (actual != expected)
    {
      support_record_failure ();
      printf ("error: unexpected result for for %s%%%s\n",
              address, scope);
      printf ("  expected: %" PRIu32 "\n", expected);
      printf ("  actual:   %" PRIu32 "\n", actual);
    }

  struct sockaddr_in6 sa;
  memset (&sa, 0xc0, sizeof (sa));
  if (call_gai (AF_UNSPEC, address, scope, &sa))
    check_ai ("AF_UNSPEC", address, scope, &sa, &addr, expected);
  else
    {
      support_record_failure ();
      printf ("error: unexpected getaddrinfo failure for %s%%%s (AF_UNSPEC)\n",
              address, scope);
    }
  memset (&sa, 0xc0, sizeof (sa));
  if (call_gai (AF_INET6, address, scope, &sa))
    check_ai ("AF_INET6", address, scope, &sa, &addr, expected);
  else
    {
      support_record_failure ();
      printf ("error: unexpected getaddrinfo failure for %s%%%s (AF_INET6)\n",
              address, scope);
    }
}

static int
do_test (void)
{
  setup_interface ();

  static const char *test_addresses[]
    = { "::", "::1", "2001:db8::1", NULL };
  for (int i = 0; test_addresses[i] != NULL; ++i)
    {
      expect_success (test_addresses[i], "0", 0);
      expect_success (test_addresses[i], "5555", 5555);

      expect_failure (test_addresses[i], "");
      expect_failure (test_addresses[i], "-1");
      expect_failure (test_addresses[i], "-99");
      expect_failure (test_addresses[i], "037777777777");
      expect_failure (test_addresses[i], "0x");
      expect_failure (test_addresses[i], "0x1");
    }

  if (interface_name != NULL)
    {
      expect_success ("fe80::1", interface_name, interface_index);
      expect_success ("ff02::1", interface_name, interface_index);
      expect_success ("ff01::1", interface_name, interface_index);
      expect_failure ("::", interface_name);
      expect_failure ("::1", interface_name);
      expect_failure ("2001:db8::1", interface_name);
    }

  return 0;
}

#include <support/test-driver.c>
