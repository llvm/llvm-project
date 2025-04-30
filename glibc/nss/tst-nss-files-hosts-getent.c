/* Enumerate /etc/hosts with a long line (bug 18991).
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


#include <dlfcn.h>
#include <errno.h>
#include <gnu/lib-names.h>
#include <netdb.h>
#include <nss.h>
#include <stdlib.h>
#include <stdio.h>
#include <support/check.h>
#include <support/check_nss.h>
#include <support/namespace.h>
#include <support/support.h>
#include <support/test-driver.h>
#include <support/xmemstream.h>
#include <support/xstdio.h>
#include <support/xunistd.h>

struct support_chroot *chroot_env;

/* Number of alias names in the long line.  This is varied to catch
   different cases where the ERANGE handling can go wrong (line buffer
   length, alias buffer).  */
static int name_count;

/* Write /etc/hosts, from outside of the chroot.  */
static void
write_hosts (void)
{
  FILE *fp = xfopen (chroot_env->path_hosts, "w");
  fputs ("127.0.0.1   localhost localhost.localdomain\n", fp);
  fputs ("192.0.2.2 host2.example.com\n", fp);
  fputs ("192.0.2.1", fp);
  for (int i = 0; i < name_count; ++i)
    fprintf (fp, " host%d.example.com", i);
  fputs ("\n192.0.2.80 www.example.com\n"
         "192.0.2.5 host5.example.com\n"
         "192.0.2.81 www1.example.com\n", fp);
  xfclose (fp);
}

const char *host1_expected =
  "name: localhost\n"
  "alias: localhost.localdomain\n"
  "address: 127.0.0.1\n";
const char *host2_expected =
  "name: host2.example.com\n"
  "address: 192.0.2.2\n";
const char *host4_expected =
  "name: www.example.com\n"
  "address: 192.0.2.80\n";
const char *host5_expected =
  "name: host5.example.com\n"
  "address: 192.0.2.5\n";
const char *host6_expected =
  "name: www1.example.com\n"
  "address: 192.0.2.81\n";

static void
prepare (int argc, char **argv)
{
  chroot_env = support_chroot_create
    ((struct support_chroot_configuration)
     {
       .resolv_conf = "",
       .hosts = "",             /* Filled in by write_hosts.  */
       .host_conf = "multi on\n",
     });
}

/* If -1, no sethostent call.  Otherwise, pass do_stayopen as the
   sethostent argument.  */
static int do_stayopen;

/* If non-zero, perform an endostent call.  */
static int do_endent;

static void
subprocess_getent (void *closure)
{
  xchroot (chroot_env->path_chroot);

  errno = 0;
  if (do_stayopen >= 0)
    sethostent (do_stayopen);
  TEST_VERIFY (errno == 0);

  int i = 0;
  while (true)
    {
      struct xmemstream expected;
      xopen_memstream (&expected);
      switch (++i)
        {
        case 1:
          fputs (host1_expected, expected.out);
          break;
        case 2:
          fputs (host2_expected, expected.out);
          break;
        case 3:
          fputs ("name: host0.example.com\n", expected.out);
          for (int j = 1; j < name_count; ++j)
            fprintf (expected.out, "alias: host%d.example.com\n", j);
          fputs ("address: 192.0.2.1\n", expected.out);
          break;
        case 4:
          fputs (host4_expected, expected.out);
          break;
        case 5:
          fputs (host5_expected, expected.out);
          break;
        case 6:
          fputs (host6_expected, expected.out);
          break;
        default:
          fprintf (expected.out, "*** unexpected host %d ***\n", i);
          break;
        }
      xfclose_memstream (&expected);
      char *context = xasprintf ("do_stayopen=%d host=%d", do_stayopen, i);

      errno = 0;
      struct hostent *e = gethostent ();
      if (e == NULL)
        {
          TEST_VERIFY (errno == 0);
          break;
        }
      check_hostent (context, e, expected.buffer);
      free (context);
      free (expected.buffer);
    }

  errno = 0;
  if (do_endent)
    endhostent ();
  TEST_VERIFY (errno == 0);

  /* Exercise process termination.   */
  exit (0);
}

/* getaddrinfo test.  To be run from a subprocess.  */
static void
test_gai (int family)
{
  struct addrinfo hints =
    {
      .ai_family = family,
      .ai_protocol = IPPROTO_TCP,
      .ai_socktype = SOCK_STREAM,
    };

  struct addrinfo *ai;
  int ret = getaddrinfo ("host2.example.com", "80", &hints, &ai);
  check_addrinfo ("host2.example.com", ai, ret,
                  "address: STREAM/TCP 192.0.2.2 80\n"
                  "address: STREAM/TCP 192.0.2.1 80\n");

  ret = getaddrinfo ("host5.example.com", "80", &hints, &ai);
  check_addrinfo ("host5.example.com", ai, ret,
                  "address: STREAM/TCP 192.0.2.1 80\n"
                  "address: STREAM/TCP 192.0.2.5 80\n");

  ret = getaddrinfo ("www.example.com", "80", &hints, &ai);
  check_addrinfo ("www.example.com", ai, ret,
                  "address: STREAM/TCP 192.0.2.80 80\n");

  ret = getaddrinfo ("www1.example.com", "80", &hints, &ai);
  check_addrinfo ("www1.example.com", ai, ret,
                  "address: STREAM/TCP 192.0.2.81 80\n");
}

/* Subprocess routine for gethostbyname/getaddrinfo testing.  */
static void
subprocess_gethost (void *closure)
{
  xchroot (chroot_env->path_chroot);

  /* This tests enlarging the read buffer in the multi case.  */
  struct xmemstream expected;
  xopen_memstream (&expected);
  fputs ("name: host2.example.com\n", expected.out);
  for (int j = 1; j < name_count; ++j)
    /* NB: host2 is duplicated in the alias list.  */
    fprintf (expected.out, "alias: host%d.example.com\n", j);
  fputs ("alias: host0.example.com\n"
         "address: 192.0.2.2\n"
         "address: 192.0.2.1\n",
         expected.out);
  xfclose_memstream (&expected);
  check_hostent ("host2.example.com",
                 gethostbyname ("host2.example.com"),
                 expected.buffer);
  free (expected.buffer);

  /* Similarly, but with a different order in the /etc/hosts file.  */
  xopen_memstream (&expected);
  fputs ("name: host0.example.com\n", expected.out);
  for (int j = 1; j < name_count; ++j)
    fprintf (expected.out, "alias: host%d.example.com\n", j);
  /* NB: host5 is duplicated in the alias list.  */
  fputs ("alias: host5.example.com\n"
         "address: 192.0.2.1\n"
         "address: 192.0.2.5\n",
         expected.out);
  xfclose_memstream (&expected);
  check_hostent ("host5.example.com",
                 gethostbyname ("host5.example.com"),
                 expected.buffer);
  free (expected.buffer);

  check_hostent ("www.example.com",
                 gethostbyname ("www.example.com"),
                 host4_expected);
  check_hostent ("www1.example.com",
                 gethostbyname ("www1.example.com"),
                 host6_expected);

  test_gai (AF_INET);
  test_gai (AF_UNSPEC);
}

static int
do_test (void)
{
  support_become_root ();
  if (!support_can_chroot ())
    return EXIT_UNSUPPORTED;

  __nss_configure_lookup ("hosts", "files");
  if (dlopen (LIBNSS_FILES_SO, RTLD_LAZY) == NULL)
    FAIL_EXIT1 ("could not load " LIBNSS_DNS_SO ": %s", dlerror ());

  /* Each name takes about 20 bytes, so this covers a wide range of
     buffer sizes, from less than 1000 bytes to about 18000 bytes.  */
  for (name_count = 40; name_count <= 850; ++name_count)
    {
      write_hosts ();

      for (do_stayopen = -1; do_stayopen < 2; ++do_stayopen)
        for (do_endent = 0; do_endent < 2; ++do_endent)
          {
            if (test_verbose > 0)
              printf ("info: name_count=%d do_stayopen=%d do_endent=%d\n",
                      name_count, do_stayopen, do_endent);
            support_isolate_in_subprocess (subprocess_getent, NULL);
          }

      support_isolate_in_subprocess (subprocess_gethost, NULL);
    }

  support_chroot_free (chroot_env);
  return 0;
}

#define PREPARE prepare
#include <support/test-driver.c>
