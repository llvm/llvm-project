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
 * authdes_prot.c, XDR routines for DES authentication
 */

#include <rpc/types.h>
#include <rpc/xdr.h>
#include <rpc/auth.h>
#include <rpc/auth_des.h>
#include <shlib-compat.h>

#define ATTEMPT(xdr_op) if (!(xdr_op)) return (FALSE)

bool_t
xdr_authdes_cred (XDR *xdrs, struct authdes_cred *cred)
{
  /*
   * Unrolled xdr
   */
  ATTEMPT (xdr_enum (xdrs, (enum_t *) & cred->adc_namekind));
  switch (cred->adc_namekind)
    {
    case ADN_FULLNAME:
      ATTEMPT (xdr_string (xdrs, &cred->adc_fullname.name, MAXNETNAMELEN));
      ATTEMPT (xdr_opaque (xdrs, (caddr_t) & cred->adc_fullname.key,
			   sizeof (des_block)));
      ATTEMPT (xdr_opaque (xdrs, (caddr_t) & cred->adc_fullname.window,
			   sizeof (cred->adc_fullname.window)));
      return (TRUE);
    case ADN_NICKNAME:
      ATTEMPT (xdr_opaque (xdrs, (caddr_t) & cred->adc_nickname,
			   sizeof (cred->adc_nickname)));
      return TRUE;
    default:
      return FALSE;
    }
}
libc_hidden_nolink_sunrpc (xdr_authdes_cred, GLIBC_2_1)


bool_t
xdr_authdes_verf (register XDR *xdrs, register struct authdes_verf *verf)
{
  /*
   * Unrolled xdr
   */
  ATTEMPT (xdr_opaque (xdrs, (caddr_t) & verf->adv_xtimestamp,
		       sizeof (des_block)));
  ATTEMPT (xdr_opaque (xdrs, (caddr_t) & verf->adv_int_u,
		       sizeof (verf->adv_int_u)));
  return TRUE;
}
libc_hidden_nolink_sunrpc (xdr_authdes_verf, GLIBC_2_1)
