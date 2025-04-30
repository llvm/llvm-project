/* Test static linking against multiple libraries, to find symbol conflicts.
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

#include <math.h>
#include <pthread.h>
#if USE_CRYPT
# include <crypt.h>
#endif
#include <resolv.h>
#include <dlfcn.h>
#include <utmp.h>
#include <aio.h>
#include <netdb.h>

/* These references force linking the executable against central
   functions in the static libraries, pulling significant parts of
   each library into the link.  */
void *references[] =
  {
    &pow,                       /* libm */
    &pthread_create,            /* libpthread */
#if USE_CRYPT
    &crypt,                     /* libcrypt */
#endif
    &res_send,                  /* libresolv */
    &dlopen,                    /* libdl */
    &login,                     /* libutil */
    &aio_init,                  /* librt */
    &getaddrinfo_a,             /* libanl */
  };

static int
do_test (void)
{
  /* This is a link-time test.  There is nothing to run here.  */
  return 0;
}

#include <support/test-driver.c>
