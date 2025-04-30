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
 * authunix_prot.c
 * XDR for UNIX style authentication parameters for RPC
 */

#include <rpc/types.h>
#include <rpc/xdr.h>
#include <rpc/auth.h>
#include <rpc/auth_unix.h>
#include <shlib-compat.h>

/*
 * XDR for unix authentication parameters.
 * Unfortunately, none of these can be declared const.
 */
bool_t
xdr_authunix_parms (XDR * xdrs, struct authunix_parms *p)
{
  if (xdr_u_long (xdrs, &(p->aup_time))
      && xdr_string (xdrs, &(p->aup_machname), MAX_MACHINE_NAME)
      && (sizeof (uid_t) == sizeof (short int)
	  ? xdr_u_short (xdrs, (u_short *) & (p->aup_uid))
	  : xdr_u_int (xdrs, (u_int *) & (p->aup_uid)))
      && (sizeof (gid_t) == sizeof (short int)
	  ? xdr_u_short (xdrs, (u_short *) & (p->aup_gid))
	  : xdr_u_int (xdrs, (u_int *) & (p->aup_gid)))
      && xdr_array (xdrs, (caddr_t *) & (p->aup_gids),
		    & (p->aup_len), NGRPS, sizeof (gid_t),
		    (sizeof (gid_t) == sizeof (short int)
		     ? (xdrproc_t) xdr_u_short
		     : (xdrproc_t) xdr_u_int)))
    {
      return TRUE;
    }
  return FALSE;
}
libc_hidden_nolink_sunrpc (xdr_authunix_parms, GLIBC_2_0)
