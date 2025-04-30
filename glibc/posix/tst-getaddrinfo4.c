/* Test getaddrinfo return value, [BZ #15339].
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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

#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <netdb.h>

static int
try (const char *service, int family, int flags)
{
  struct addrinfo hints, *h, *ai;
  int res;

  memset (&hints, 0, sizeof hints);
  hints.ai_family = family;
  hints.ai_flags = flags;

  errno = 0;
  h = (family || flags) ? &hints : NULL;
  res = getaddrinfo ("example.net", service, h, &ai);
  switch (res)
    {
    case 0:
    case EAI_AGAIN:
    case EAI_NONAME:
      printf ("SUCCESS getaddrinfo(service=%s, family=%d, flags=%d): %s: %m\n",
              service ?: "NULL", family, flags, gai_strerror (res));
      return 0;
    }
  printf ("FAIL getaddrinfo(service=%s, family=%d, flags=%d): %s: %m\n",
          service ?: "NULL", family, flags, gai_strerror (res));
  return 1;
}

static int
do_test (void)
{
  int err = 0;
  err |= try (NULL, 0, 0);
  err |= try (NULL, AF_UNSPEC, AI_ADDRCONFIG);
  err |= try (NULL, AF_INET, 0);
  err |= try (NULL, AF_INET6, 0);
  err |= try ("http", 0, 0);
  err |= try ("http", AF_UNSPEC, AI_ADDRCONFIG);
  err |= try ("http", AF_INET, 0);
  err |= try ("http", AF_INET6, 0);
  return err;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
