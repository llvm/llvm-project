/* Parse /etc/hosts in multi mode with a trailing long line (bug 21915).
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
#include <support/check.h>
#include <support/check_nss.h>
#include <support/namespace.h>
#include <support/test-driver.h>
#include <support/xunistd.h>

struct support_chroot *chroot_env;

#define X10 "XXXXXXXXXX"
#define X100 X10 X10 X10 X10 X10 X10 X10 X10 X10 X10
#define X1000 X100 X100 X100 X100 X100 X100 X100 X100 X100 X100

static void
prepare (int argc, char **argv)
{
  chroot_env = support_chroot_create
    ((struct support_chroot_configuration)
     {
       .resolv_conf = "",
       .hosts =
         "127.0.0.1   localhost localhost.localdomain\n"
         "::1         localhost localhost.localdomain\n"
         "192.0.2.1   example.com\n"
         "#" X1000 X100 "\n",
       .host_conf = "multi on\n",
     });
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

  xchroot (chroot_env->path_chroot);

  errno = ERANGE;
  h_errno = NETDB_INTERNAL;
  check_hostent ("gethostbyname example.com",
                 gethostbyname ("example.com"),
                 "name: example.com\n"
                 "address: 192.0.2.1\n");
  errno = ERANGE;
  h_errno = NETDB_INTERNAL;
  check_hostent ("gethostbyname2 AF_INET example.com",
                 gethostbyname2 ("example.com", AF_INET),
                 "name: example.com\n"
                 "address: 192.0.2.1\n");
  {
    struct addrinfo hints =
      {
        .ai_family = AF_UNSPEC,
        .ai_socktype = SOCK_STREAM,
        .ai_protocol = IPPROTO_TCP,
      };
    errno = ERANGE;
    h_errno = NETDB_INTERNAL;
    struct addrinfo *ai;
    int ret = getaddrinfo ("example.com", "80", &hints, &ai);
    check_addrinfo ("example.com AF_UNSPEC", ai, ret,
                    "address: STREAM/TCP 192.0.2.1 80\n");
    if (ret == 0)
      freeaddrinfo (ai);

    hints.ai_family = AF_INET;
    errno = ERANGE;
    h_errno = NETDB_INTERNAL;
    ret = getaddrinfo ("example.com", "80", &hints, &ai);
    check_addrinfo ("example.com AF_INET", ai, ret,
                    "address: STREAM/TCP 192.0.2.1 80\n");
    if (ret == 0)
      freeaddrinfo (ai);
  }

  support_chroot_free (chroot_env);
  return 0;
}

#define PREPARE prepare
#include <support/test-driver.c>
