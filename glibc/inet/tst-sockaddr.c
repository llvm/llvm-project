/* Tests for socket address type definitions.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

#include <netinet/in.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>

/* This is a copy of the previous definition of struct
   sockaddr_storage.  It is not equal to the old value of _SS_SIZE
   (128) on all architectures.  We must stay compatible with the old
   definition.  */

#define OLD_REFERENCE_SIZE 128
#define OLD_PADSIZE (OLD_REFERENCE_SIZE - (2 * sizeof (__ss_aligntype)))
struct sockaddr_storage_old
  {
    __SOCKADDR_COMMON (old_);
    __ss_aligntype old_align;
    char old_padding[OLD_PADSIZE];
  };

static bool errors;

static void
check (bool ok, const char *message)
{
  if (!ok)
    {
      printf ("error: failed check: %s\n", message);
      errors = true;
    }
}

static int
do_test (void)
{
  check (OLD_REFERENCE_SIZE >= _SS_SIZE,
         "old target size is not smaller than actual size");
  check (sizeof (struct sockaddr_storage_old)
         == sizeof (struct sockaddr_storage),
         "old and new sizes match");
  check (__alignof (struct sockaddr_storage_old)
         == __alignof (struct sockaddr_storage),
         "old and new alignment matches");
  check (offsetof (struct sockaddr_storage_old, old_family)
         == offsetof (struct sockaddr_storage, ss_family),
         "old and new family offsets match");
  check (sizeof (struct sockaddr_storage) == _SS_SIZE,
         "struct sockaddr_storage size");

  /* Check for lack of holes in the struct definition.   */
  check (offsetof (struct sockaddr_storage, __ss_padding)
         == __SOCKADDR_COMMON_SIZE,
         "implicit padding before explicit padding");
  check (offsetof (struct sockaddr_storage, __ss_align)
         == __SOCKADDR_COMMON_SIZE
           + sizeof (((struct sockaddr_storage) {}).__ss_padding),
         "implicit padding before explicit padding");

  /* Check for POSIX compatibility requirements between struct
     sockaddr_storage and struct sockaddr_un.  */
  check (sizeof (struct sockaddr_storage) >= sizeof (struct sockaddr_un),
         "sockaddr_storage is at least as large as sockaddr_un");
  check (__alignof (struct sockaddr_storage)
         >= __alignof (struct sockaddr_un),
         "sockaddr_storage is at least as aligned as sockaddr_un");
  check (offsetof (struct sockaddr_storage, ss_family)
         == offsetof (struct sockaddr_un, sun_family),
         "family offsets match");

  /* Check that the compiler preserves bit patterns in aggregate
     copies.  Based on <https://gcc.gnu.org/PR71120>.  */
  check (sizeof (struct sockaddr_storage) >= sizeof (struct sockaddr_in),
         "sockaddr_storage is at least as large as sockaddr_in");
  {
    struct sockaddr_storage addr;
    memset (&addr, 0, sizeof (addr));
    {
      struct sockaddr_in *sinp = (struct sockaddr_in *)&addr;
      sinp->sin_family = AF_INET;
      sinp->sin_addr.s_addr = htonl (INADDR_LOOPBACK);
      sinp->sin_port = htons (80);
    }
    struct sockaddr_storage copy;
    copy = addr;

    struct sockaddr_storage *p = malloc (sizeof (*p));
    if (p == NULL)
      {
        printf ("error: malloc: %m\n");
        return 1;
      }
    *p = copy;
    const struct sockaddr_in *sinp = (const struct sockaddr_in *)p;
    check (sinp->sin_family == AF_INET, "sin_family");
    check (sinp->sin_addr.s_addr == htonl (INADDR_LOOPBACK), "sin_addr");
    check (sinp->sin_port == htons (80), "sin_port");
    free (p);
  }

  return errors;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
