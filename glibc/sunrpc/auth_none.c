/*
 * Copyright (c) 2010, Oracle America, Inc.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the "Oracle America, Inc." nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *   FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *   COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 *   INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 *   GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 *   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
/*
 * auth_none.c
 * Creates a client authentication handle for passing "null"
 * credentials and verifiers to remote systems.
 */

#include <rpc/rpc.h>
#include <libc-lock.h>
#include <shlib-compat.h>

#define MAX_MARSHAL_SIZE 20

/*
 * Authenticator operations routines
 */
static void authnone_verf (AUTH *);
static void authnone_destroy (AUTH *);
static bool_t authnone_marshal (AUTH *, XDR *);
static bool_t authnone_validate (AUTH *, struct opaque_auth *);
static bool_t authnone_refresh (AUTH *);

static const struct auth_ops ops = {
  authnone_verf,
  authnone_marshal,
  authnone_validate,
  authnone_refresh,
  authnone_destroy
};

/* Internal data and routines */

struct authnone_private_s {
  AUTH no_client;
  char marshalled_client[MAX_MARSHAL_SIZE];
  u_int mcnt;
};

static struct authnone_private_s authnone_private;
__libc_once_define(static, authnone_private_guard);

static void authnone_create_once (void);

static void
authnone_create_once (void)
{
  struct authnone_private_s *ap;
  XDR xdr_stream;
  XDR *xdrs;

  ap = &authnone_private;

  ap->no_client.ah_cred = ap->no_client.ah_verf = _null_auth;
  ap->no_client.ah_ops = (struct auth_ops *) &ops;
  xdrs = &xdr_stream;
  xdrmem_create (xdrs, ap->marshalled_client,
		 (u_int) MAX_MARSHAL_SIZE, XDR_ENCODE);
  (void) xdr_opaque_auth (xdrs, &ap->no_client.ah_cred);
  (void) xdr_opaque_auth (xdrs, &ap->no_client.ah_verf);
  ap->mcnt = XDR_GETPOS (xdrs);
  XDR_DESTROY (xdrs);
}

AUTH *
authnone_create (void)
{
  __libc_once (authnone_private_guard, authnone_create_once);
  return &authnone_private.no_client;
}
libc_hidden_nolink_sunrpc (authnone_create, GLIBC_2_0)

static bool_t
authnone_marshal (AUTH *client, XDR *xdrs)
{
  struct authnone_private_s *ap;

  /* authnone_create returned authnone_private->no_client, which is
     the first field of struct authnone_private_s.  */
  ap = (struct authnone_private_s *) client;
  if (ap == NULL)
    return FALSE;
  return (*xdrs->x_ops->x_putbytes) (xdrs, ap->marshalled_client, ap->mcnt);
}

static void
authnone_verf (AUTH *auth)
{
}

static bool_t
authnone_validate (AUTH *auth, struct opaque_auth *oa)
{
  return TRUE;
}

static bool_t
authnone_refresh (AUTH *auth)
{
  return FALSE;
}

static void
authnone_destroy (AUTH *auth)
{
}
