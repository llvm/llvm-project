/* Copyright (c) 2010, Oracle America, Inc.
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

#include <rpc/key_prot.h>
#include <shlib-compat.h>

bool_t
xdr_keystatus (XDR * xdrs, keystatus * objp)
{
  if (!xdr_enum (xdrs, (enum_t *) objp))
    return FALSE;

  return TRUE;
}
libc_hidden_nolink_sunrpc (xdr_keystatus, GLIBC_2_0)

bool_t
xdr_keybuf (XDR * xdrs, keybuf objp)
{
  if (!xdr_opaque (xdrs, objp, HEXKEYBYTES))
    return FALSE;

  return TRUE;
}
libc_hidden_nolink_sunrpc (xdr_keybuf, GLIBC_2_0)

bool_t
xdr_netnamestr (XDR * xdrs, netnamestr * objp)
{
  if (!xdr_string (xdrs, objp, MAXNETNAMELEN))
    return FALSE;

  return TRUE;
}
libc_hidden_nolink_sunrpc (xdr_netnamestr, GLIBC_2_1)

bool_t
xdr_cryptkeyarg (XDR * xdrs, cryptkeyarg * objp)
{
  if (!xdr_netnamestr (xdrs, &objp->remotename))
    return FALSE;

  if (!xdr_des_block (xdrs, &objp->deskey))
    return FALSE;

  return TRUE;
}
libc_hidden_nolink_sunrpc (xdr_cryptkeyarg, GLIBC_2_0)

bool_t
xdr_cryptkeyarg2 (XDR * xdrs, cryptkeyarg2 * objp)
{
  if (!xdr_netnamestr (xdrs, &objp->remotename))
    return FALSE;
  if (!xdr_netobj (xdrs, &objp->remotekey))
    return FALSE;
  if (!xdr_des_block (xdrs, &objp->deskey))
    return FALSE;
  return TRUE;
}
libc_hidden_nolink_sunrpc (xdr_cryptkeyarg2, GLIBC_2_0)

bool_t
xdr_cryptkeyres (XDR * xdrs, cryptkeyres * objp)
{
  if (!xdr_keystatus (xdrs, &objp->status))
    return FALSE;
  switch (objp->status)
    {
    case KEY_SUCCESS:
      if (!xdr_des_block (xdrs, &objp->cryptkeyres_u.deskey))
	return FALSE;
      break;
    default:
      break;
    }
  return TRUE;
}
libc_hidden_nolink_sunrpc (xdr_cryptkeyres, GLIBC_2_0)

bool_t
xdr_unixcred (XDR * xdrs, unixcred * objp)
{
  if (!xdr_u_int (xdrs, &objp->uid))
    return FALSE;
  if (!xdr_u_int (xdrs, &objp->gid))
    return FALSE;
  if (!xdr_array (xdrs, (void *) &objp->gids.gids_val,
		  (u_int *) & objp->gids.gids_len, MAXGIDS,
		  sizeof (u_int), (xdrproc_t) xdr_u_int))
    return FALSE;
  return TRUE;
}
libc_hidden_nolink_sunrpc (xdr_unixcred, GLIBC_2_1)

bool_t
xdr_getcredres (XDR * xdrs, getcredres * objp)
{
  if (!xdr_keystatus (xdrs, &objp->status))
    return FALSE;
  switch (objp->status)
    {
    case KEY_SUCCESS:
      if (!xdr_unixcred (xdrs, &objp->getcredres_u.cred))
	return FALSE;
      break;
    default:
      break;
    }
  return TRUE;
}
libc_hidden_nolink_sunrpc (xdr_getcredres, GLIBC_2_1)

bool_t
xdr_key_netstarg (XDR * xdrs, key_netstarg * objp)
{
  if (!xdr_keybuf (xdrs, objp->st_priv_key))
    return FALSE;
  if (!xdr_keybuf (xdrs, objp->st_pub_key))
    return FALSE;
  if (!xdr_netnamestr (xdrs, &objp->st_netname))
    return FALSE;
  return TRUE;
}
libc_hidden_nolink_sunrpc (xdr_key_netstarg, GLIBC_2_0)

bool_t
xdr_key_netstres (XDR * xdrs, key_netstres * objp)
{
  if (!xdr_keystatus (xdrs, &objp->status))
    return FALSE;
  switch (objp->status)
    {
    case KEY_SUCCESS:
      if (!xdr_key_netstarg (xdrs, &objp->key_netstres_u.knet))
	return FALSE;
      break;
    default:
      break;
    }
  return TRUE;
}
libc_hidden_nolink_sunrpc (xdr_key_netstres, GLIBC_2_0)
