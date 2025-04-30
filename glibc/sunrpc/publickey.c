/* Get public or secret key from key server.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1996.

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

#include <errno.h>
#include <rpc/netdb.h>
#include <rpc/auth_des.h>
#include <shlib-compat.h>

#include "nsswitch.h"


/* Type of the lookup function for the public key.  */
typedef int (*public_function) (const char *, char *, int *);

/* Type of the lookup function for the secret key.  */
typedef int (*secret_function) (const char *, char *, const char *, int *);

int
getpublickey (const char *name, char *key)
{
  nss_action_list nip;
  union
  {
    public_function f;
    void *ptr;
  } fct;
  enum nss_status status = NSS_STATUS_UNAVAIL;
  int no_more;

  no_more = __nss_publickey_lookup2 (&nip, "getpublickey", NULL, &fct.ptr);

  while (! no_more)
    {
      status = (*fct.f) (name, key, &errno);

      no_more = __nss_next2 (&nip, "getpublickey", NULL, &fct.ptr, status, 0);
    }

  return status == NSS_STATUS_SUCCESS;
}
libc_hidden_nolink_sunrpc (getpublickey, GLIBC_2_0)


int
getsecretkey (const char *name, char *key, const char *passwd)
{
  nss_action_list nip;
  union
  {
    secret_function f;
    void *ptr;
  } fct;
  enum nss_status status = NSS_STATUS_UNAVAIL;
  int no_more;

  no_more = __nss_publickey_lookup2 (&nip, "getsecretkey", NULL, &fct.ptr);

  while (! no_more)
    {
      status = (*fct.f) (name, key, passwd, &errno);

      no_more = __nss_next2 (&nip, "getsecretkey", NULL, &fct.ptr, status, 0);
    }

  return status == NSS_STATUS_SUCCESS;
}
libc_hidden_nolink_sunrpc (getsecretkey, GLIBC_2_0)
