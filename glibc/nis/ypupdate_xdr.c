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

#include <rpcsvc/ypupd.h>
#include <shlib-compat.h>

bool_t
xdr_yp_buf (XDR *xdrs, yp_buf *objp)
{
  return xdr_bytes (xdrs, (char **) &objp->yp_buf_val,
		    (u_int *) &objp->yp_buf_len, ~0);
}
libnsl_hidden_nolink_def (xdr_yp_buf, GLIBC_2_0)

bool_t
xdr_ypupdate_args (XDR *xdrs, ypupdate_args *objp)
{
  if (!xdr_string (xdrs, &objp->mapname, ~0))
    return FALSE;
  if (!xdr_yp_buf (xdrs, &objp->key))
    return FALSE;
  return xdr_yp_buf (xdrs, &objp->datum);
}
libnsl_hidden_nolink_def (xdr_ypupdate_args, GLIBC_2_0)

bool_t
xdr_ypdelete_args (XDR *xdrs, ypdelete_args *objp)
{
  if (!xdr_string (xdrs, &objp->mapname, ~0))
    return FALSE;
  return xdr_yp_buf (xdrs, &objp->key);
}
libnsl_hidden_nolink_def (xdr_ypdelete_args, GLIBC_2_0)
