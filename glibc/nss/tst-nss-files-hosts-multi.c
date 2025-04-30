/* Parse /etc/hosts in multi mode with many addresses/aliases.
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
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <support/check.h>
#include <support/check_nss.h>
#include <support/namespace.h>
#include <support/support.h>
#include <support/test-driver.h>
#include <support/test-driver.h>
#include <support/xmemstream.h>
#include <support/xstdio.h>
#include <support/xunistd.h>
#include <sys/resource.h>

struct support_chroot *chroot_env;

static void
prepare (int argc, char **argv)
{
  chroot_env = support_chroot_create
    ((struct support_chroot_configuration)
     {
       .resolv_conf = "",
       .hosts = "",             /* See write_hosts below.  */
       .host_conf = "multi on\n",
     });
}

/* Create the /etc/hosts file from outside the chroot.  */
static void
write_hosts (int count)
{
  TEST_VERIFY (count > 0 && count <= 65535);
  FILE *fp = xfopen (chroot_env->path_hosts, "w");
  fputs ("127.0.0.1   localhost localhost.localdomain\n"
         "::1         localhost localhost.localdomain\n",
         fp);
  for (int i = 0; i < count; ++i)
    {
      fprintf (fp, "10.4.%d.%d www4.example.com\n",
               (i / 256) & 0xff, i & 0xff);
      fprintf (fp, "10.46.%d.%d www.example.com\n",
               (i / 256) & 0xff, i & 0xff);
      fprintf (fp, "192.0.2.1 alias.example.com v4-%d.example.com\n", i);
      fprintf (fp, "2001:db8::6:%x www6.example.com\n", i);
      fprintf (fp, "2001:db8::46:%x www.example.com\n", i);
      fprintf (fp, "2001:db8::1 alias.example.com v6-%d.example.com\n", i);
    }
  xfclose (fp);
}

/* Parameters of a single test.  */
struct test_params
{
  const char *name;             /* Name to query.  */
  const char *marker;           /* Address marker for the name.  */
  int count;                    /* Number of addresses/aliases.  */
  int family;                   /* AF_INET, AF_INET_6 or AF_UNSPEC.  */
  bool canonname;               /* True if AI_CANONNAME should be enabled.  */
};

/* Expected result of gethostbyname/gethostbyname2.  */
static char *
expected_ghbn (const struct test_params *params)
{
  TEST_VERIFY (params->family == AF_INET || params->family == AF_INET6);

  struct xmemstream expected;
  xopen_memstream (&expected);
  if (strcmp (params->name, "alias.example.com") == 0)
    {
      fprintf (expected.out, "name: %s\n", params->name);
      char af;
      if (params->family == AF_INET)
        af = '4';
      else
        af = '6';
      for (int i = 0; i < params->count; ++i)
        fprintf (expected.out, "alias: v%c-%d.example.com\n", af, i);

      for (int i = 0; i < params->count; ++i)
        if (params->family == AF_INET)
          fputs ("address: 192.0.2.1\n", expected.out);
        else
          fputs ("address: 2001:db8::1\n", expected.out);
    }
  else /* www/www4/www6 name.  */
    {
      bool do_ipv4 = params->family == AF_INET
        && strncmp (params->name, "www6", 4) != 0;
      bool do_ipv6 = params->family == AF_INET6
        && strncmp (params->name, "www4", 4) != 0;
      if (do_ipv4 || do_ipv6)
        {
          fprintf (expected.out, "name: %s\n", params->name);
          if (do_ipv4)
            for (int i = 0; i < params->count; ++i)
              fprintf (expected.out, "address: 10.%s.%d.%d\n",
                       params->marker, i / 256, i % 256);
          if (do_ipv6)
            for (int i = 0; i < params->count; ++i)
              fprintf (expected.out, "address: 2001:db8::%s:%x\n",
                       params->marker, i);
        }
      else
        fputs ("error: HOST_NOT_FOUND\n", expected.out);
    }
  xfclose_memstream (&expected);
  return expected.buffer;
}

/* Expected result of getaddrinfo.  */
static char *
expected_gai (const struct test_params *params)
{
  bool do_ipv4 = false;
  bool do_ipv6 = false;
  if (params->family == AF_UNSPEC)
    do_ipv4 = do_ipv6 = true;
  else if (params->family == AF_INET)
    do_ipv4 = true;
  else if (params->family == AF_INET6)
    do_ipv6 = true;

  struct xmemstream expected;
  xopen_memstream (&expected);
  if (strcmp (params->name, "alias.example.com") == 0)
    {
      if (params->canonname)
        fprintf (expected.out,
                 "flags: AI_CANONNAME\n"
                 "canonname: %s\n",
                 params->name);

      if (do_ipv4)
        for (int i = 0; i < params->count; ++i)
          fputs ("address: STREAM/TCP 192.0.2.1 80\n", expected.out);
      if (do_ipv6)
        for (int i = 0; i < params->count; ++i)
          fputs ("address: STREAM/TCP 2001:db8::1 80\n", expected.out);
    }
  else /* www/www4/www6 name.  */
    {
      if (strncmp (params->name, "www4", 4) == 0)
        do_ipv6 = false;
      else if (strncmp (params->name, "www6", 4) == 0)
        do_ipv4 = false;
      /* Otherwise, we have www as the name, so we do both.  */

      if (do_ipv4 || do_ipv6)
        {
          if (params->canonname)
            fprintf (expected.out,
                     "flags: AI_CANONNAME\n"
                     "canonname: %s\n",
                     params->name);

          if (do_ipv4)
            for (int i = 0; i < params->count; ++i)
              fprintf (expected.out, "address: STREAM/TCP 10.%s.%d.%d 80\n",
                       params->marker, i / 256, i % 256);
          if (do_ipv6)
            for (int i = 0; i < params->count; ++i)
              fprintf (expected.out,
                       "address: STREAM/TCP 2001:db8::%s:%x 80\n",
                       params->marker, i);
        }
      else
        fputs ("error: Name or service not known\n", expected.out);
    }
  xfclose_memstream (&expected);
  return expected.buffer;
}

static void
run_gbhn_gai (struct test_params *params)
{
  char *ctx = xasprintf ("name=%s marker=%s count=%d family=%d",
                         params->name, params->marker, params->count,
                         params->family);
  if (test_verbose > 0)
    printf ("info: %s\n", ctx);

  /* Check gethostbyname, gethostbyname2.  */
  if (params->family == AF_INET)
    {
      char *expected = expected_ghbn (params);
      check_hostent (ctx, gethostbyname (params->name), expected);
      free (expected);
    }
  if (params->family != AF_UNSPEC)
    {
      char *expected = expected_ghbn (params);
      check_hostent (ctx, gethostbyname2 (params->name, params->family),
                     expected);
      free (expected);
    }

  /* Check getaddrinfo.  */
  for (int do_canonical = 0; do_canonical < 2; ++do_canonical)
    {
      params->canonname = do_canonical;
      char *expected = expected_gai (params);
      struct addrinfo hints =
        {
          .ai_family = params->family,
          .ai_socktype = SOCK_STREAM,
          .ai_protocol = IPPROTO_TCP,
        };
      if (do_canonical)
        hints.ai_flags |= AI_CANONNAME;
      struct addrinfo *ai;
      int ret = getaddrinfo (params->name, "80", &hints, &ai);
      check_addrinfo (ctx, ai, ret, expected);
      if (ret == 0)
        freeaddrinfo (ai);
      free (expected);
    }

  free (ctx);
}

/* Callback for the subprocess which runs the test in a chroot.  */
static void
subprocess (void *closure)
{
  struct test_params *params = closure;

  xchroot (chroot_env->path_chroot);

  static const int families[] = { AF_INET, AF_INET6, AF_UNSPEC, -1 };
  static const char *const names[] =
    {
      "www.example.com", "www4.example.com", "www6.example.com",
      "alias.example.com",
      NULL
    };
  static const char *const names_marker[] = { "46", "4", "6", "" };

  for (int family_idx = 0; families[family_idx] >= 0; ++family_idx)
    {
      params->family = families[family_idx];
      for (int names_idx = 0; names[names_idx] != NULL; ++names_idx)
        {
          params->name = names[names_idx];
          params->marker = names_marker[names_idx];
          run_gbhn_gai (params);
        }
    }
}

/* Run the test for a specific number of addresses/aliases.  */
static void
run_test (int count)
{
  write_hosts (count);

  struct test_params params =
    {
      .count = count,
    };

  support_isolate_in_subprocess (subprocess, &params);
}

static int
do_test (void)
{
  support_become_root ();
  if (!support_can_chroot ())
    return EXIT_UNSUPPORTED;

  /* This test should not use gigabytes of memory.   */
  {
    struct rlimit limit;
    if (getrlimit (RLIMIT_AS, &limit) != 0)
      {
        printf ("getrlimit (RLIMIT_AS) failed: %m\n");
        return 1;
      }
    long target = 200 * 1024 * 1024;
    if (limit.rlim_cur == RLIM_INFINITY || limit.rlim_cur > target)
      {
        limit.rlim_cur = target;
        if (setrlimit (RLIMIT_AS, &limit) != 0)
          {
            printf ("setrlimit (RLIMIT_AS) failed: %m\n");
            return 1;
          }
      }
  }

  __nss_configure_lookup ("hosts", "files");
  if (dlopen (LIBNSS_FILES_SO, RTLD_LAZY) == NULL)
    FAIL_EXIT1 ("could not load " LIBNSS_DNS_SO ": %s", dlerror ());

  /* Run the tests with a few different address/alias counts.  */
  for (int count = 1; count <= 111; ++count)
    run_test (count);
  run_test (1111);
  run_test (22222);

  support_chroot_free (chroot_env);
  return 0;
}

#define TIMEOUT 40
#define PREPARE prepare
#include <support/test-driver.c>
