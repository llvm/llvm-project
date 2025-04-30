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

#include <rpcsvc/yp.h>
#include <rpcsvc/ypclnt.h>
#include <shlib-compat.h>

/* The NIS v2 protocol suggests 1024 bytes as a maximum length of all fields.
   Current Linux systems don't use this limit. To remain compatible with
   recent Linux systems we choose limits large enough to load large key and
   data values, but small enough to not pose a DoS threat. */

#define XDRMAXNAME 1024
#define XDRMAXRECORD (16 * 1024 * 1024)

bool_t
xdr_ypstat (XDR *xdrs, ypstat *objp)
{
  return xdr_enum (xdrs, (enum_t *) objp);
}
libnsl_hidden_nolink_def (xdr_ypstat, GLIBC_2_0)

bool_t
xdr_ypxfrstat (XDR *xdrs, ypxfrstat *objp)
{
  return xdr_enum (xdrs, (enum_t *) objp);
}
libnsl_hidden_nolink_def (xdr_ypxfrstat, GLIBC_2_0)

bool_t
xdr_domainname (XDR *xdrs, domainname *objp)
{
  return xdr_string (xdrs, objp, XDRMAXNAME);
}
libnsl_hidden_nolink_def (xdr_domainname, GLIBC_2_0)

bool_t
xdr_mapname (XDR *xdrs, mapname *objp)
{
  return xdr_string (xdrs, objp, XDRMAXNAME);
}
libnsl_hidden_nolink_def (xdr_mapname, GLIBC_2_0)

bool_t
xdr_peername (XDR *xdrs, peername *objp)
{
  return xdr_string (xdrs, objp, XDRMAXNAME);
}
libnsl_hidden_nolink_def (xdr_peername, GLIBC_2_0)

bool_t
xdr_keydat (XDR *xdrs, keydat *objp)
{
  return xdr_bytes (xdrs, (char **) &objp->keydat_val,
		    (u_int *) &objp->keydat_len, XDRMAXRECORD);
}
libnsl_hidden_nolink_def (xdr_keydat, GLIBC_2_0)

bool_t
xdr_valdat (XDR *xdrs, valdat *objp)
{
  return xdr_bytes (xdrs, (char **) &objp->valdat_val,
		    (u_int *) &objp->valdat_len, XDRMAXRECORD);
}
libnsl_hidden_nolink_def (xdr_valdat, GLIBC_2_0)

bool_t
xdr_ypmap_parms (XDR *xdrs, ypmap_parms *objp)
{
  if (!xdr_domainname (xdrs, &objp->domain))
    return FALSE;
  if (!xdr_mapname (xdrs, &objp->map))
    return FALSE;
  if (!xdr_u_int (xdrs, &objp->ordernum))
    return FALSE;
  return xdr_peername (xdrs, &objp->peer);
}
libnsl_hidden_nolink_def (xdr_ypmap_parms, GLIBC_2_0)

bool_t
xdr_ypreq_key (XDR *xdrs, ypreq_key *objp)
{
  if (!xdr_domainname (xdrs, &objp->domain))
    return FALSE;
  if (!xdr_mapname (xdrs, &objp->map))
    return FALSE;
  return xdr_keydat (xdrs, &objp->key);
}
libnsl_hidden_nolink_def (xdr_ypreq_key, GLIBC_2_0)

bool_t
xdr_ypreq_nokey (XDR *xdrs, ypreq_nokey *objp)
{
  if (!xdr_domainname (xdrs, &objp->domain))
    return FALSE;
  return xdr_mapname (xdrs, &objp->map);
}
libnsl_hidden_nolink_def (xdr_ypreq_nokey, GLIBC_2_0)

bool_t
xdr_ypreq_xfr (XDR *xdrs, ypreq_xfr *objp)
{
  if (!xdr_ypmap_parms (xdrs, &objp->map_parms))
    return FALSE;
  if (!xdr_u_int (xdrs, &objp->transid))
    return FALSE;
  if (!xdr_u_int (xdrs, &objp->prog))
    return FALSE;
  return xdr_u_int (xdrs, &objp->port);
}
libnsl_hidden_nolink_def (xdr_ypreq_xfr, GLIBC_2_0)

bool_t
xdr_ypresp_val (XDR *xdrs, ypresp_val *objp)
{
  if (!xdr_ypstat (xdrs, &objp->stat))
    return FALSE;
  return xdr_valdat (xdrs, &objp->val);
}
libnsl_hidden_nolink_def (xdr_ypresp_val, GLIBC_2_0)

bool_t
xdr_ypresp_key_val (XDR *xdrs, ypresp_key_val *objp)
{
  if (!xdr_ypstat (xdrs, &objp->stat))
    return FALSE;
  if (!xdr_valdat (xdrs, &objp->val))
    return FALSE;
  return xdr_keydat (xdrs, &objp->key);
}
libnsl_hidden_nolink_def (xdr_ypresp_key_val, GLIBC_2_0)

bool_t
xdr_ypresp_master (XDR *xdrs, ypresp_master *objp)
{
  if (!xdr_ypstat (xdrs, &objp->stat))
    return FALSE;
  return xdr_peername (xdrs, &objp->peer);
}
libnsl_hidden_nolink_def (xdr_ypresp_master, GLIBC_2_0)

bool_t
xdr_ypresp_order (XDR *xdrs, ypresp_order *objp)
{
  if (!xdr_ypstat (xdrs, &objp->stat))
    return FALSE;
  return xdr_u_int (xdrs, &objp->ordernum);
}
libnsl_hidden_nolink_def (xdr_ypresp_order, GLIBC_2_0)

bool_t
xdr_ypresp_all (XDR *xdrs, ypresp_all *objp)
{
  if (!xdr_bool (xdrs, &objp->more))
    return FALSE;
  switch (objp->more)
    {
    case TRUE:
      return xdr_ypresp_key_val (xdrs, &objp->ypresp_all_u.val);
    case FALSE:
      break;
    default:
      return FALSE;
    }
  return TRUE;
}
libnsl_hidden_nolink_def (xdr_ypresp_all, GLIBC_2_0)

bool_t
xdr_ypresp_xfr (XDR *xdrs, ypresp_xfr *objp)
{
  if (!xdr_u_int (xdrs, &objp->transid))
    return FALSE;
  return xdr_ypxfrstat (xdrs, &objp->xfrstat);
}
libnsl_hidden_nolink_def (xdr_ypresp_xfr, GLIBC_2_0)

bool_t
xdr_ypmaplist (XDR *xdrs, ypmaplist *objp)
{
  if (!xdr_mapname (xdrs, &objp->map))
    return FALSE;
  /* Prevent gcc warning about alias violation.  */
  char **tp = (void *) &objp->next;
  return xdr_pointer (xdrs, tp, sizeof (ypmaplist), (xdrproc_t) xdr_ypmaplist);
}
libnsl_hidden_nolink_def (xdr_ypmaplist, GLIBC_2_0)

bool_t
xdr_ypresp_maplist (XDR *xdrs, ypresp_maplist *objp)
{
  if (!xdr_ypstat (xdrs, &objp->stat))
    return FALSE;
  /* Prevent gcc warning about alias violation.  */
  char **tp = (void *) &objp->maps;
  return xdr_pointer (xdrs, tp, sizeof (ypmaplist), (xdrproc_t) xdr_ypmaplist);
}
libnsl_hidden_nolink_def (xdr_ypresp_maplist, GLIBC_2_0)

bool_t
xdr_yppush_status (XDR *xdrs, yppush_status *objp)
{
  return xdr_enum (xdrs, (enum_t *) objp);
}
libnsl_hidden_nolink_def (xdr_yppush_status, GLIBC_2_0)

bool_t
xdr_yppushresp_xfr (XDR *xdrs, yppushresp_xfr *objp)
{
  if (!xdr_u_int (xdrs, &objp->transid))
    return FALSE;
  return xdr_yppush_status (xdrs, &objp->status);
}
libnsl_hidden_nolink_def (xdr_yppushresp_xfr, GLIBC_2_0)

bool_t
xdr_ypbind_resptype (XDR *xdrs, ypbind_resptype *objp)
{
  return xdr_enum (xdrs, (enum_t *) objp);
}
libnsl_hidden_nolink_def (xdr_ypbind_resptype, GLIBC_2_0)

bool_t
xdr_ypbind_binding (XDR *xdrs, ypbind_binding *objp)
{
  if (!xdr_opaque (xdrs, objp->ypbind_binding_addr, 4))
    return FALSE;
  return xdr_opaque (xdrs, objp->ypbind_binding_port, 2);
}
libnsl_hidden_nolink_def (xdr_ypbind_binding, GLIBC_2_0)

bool_t
xdr_ypbind_resp (XDR *xdrs, ypbind_resp *objp)
{
  if (!xdr_ypbind_resptype (xdrs, &objp->ypbind_status))
    return FALSE;
  switch (objp->ypbind_status)
    {
    case YPBIND_FAIL_VAL:
      return xdr_u_int (xdrs, &objp->ypbind_resp_u.ypbind_error);
    case YPBIND_SUCC_VAL:
      return xdr_ypbind_binding (xdrs, &objp->ypbind_resp_u.ypbind_bindinfo);
    }
  return FALSE;
}
libnsl_hidden_nolink_def (xdr_ypbind_resp, GLIBC_2_0)

bool_t
xdr_ypbind_setdom (XDR *xdrs, ypbind_setdom *objp)
{
  if (!xdr_domainname (xdrs, &objp->ypsetdom_domain))
    return FALSE;
  if (!xdr_ypbind_binding (xdrs, &objp->ypsetdom_binding))
    return FALSE;
  return xdr_u_int (xdrs, &objp->ypsetdom_vers);
}
libnsl_hidden_nolink_def (xdr_ypbind_setdom, GLIBC_2_0)

bool_t
xdr_ypall(XDR *xdrs, struct ypall_callback *incallback)
{
    struct ypresp_key_val out;
    char key[YPMAXRECORD], val[YPMAXRECORD];

    /*
     * Set up key/val struct to be used during the transaction.
     */
    memset(&out, 0, sizeof out);
    out.key.keydat_val = key;
    out.key.keydat_len = sizeof(key);
    out.val.valdat_val = val;
    out.val.valdat_len = sizeof(val);

    for (;;) {
	bool_t more, status;

	/* Values pending? */
	if (!xdr_bool(xdrs, &more))
	    return FALSE;           /* can't tell! */
	if (!more)
	    return TRUE;            /* no more */

	/* Transfer key/value pair. */
	status = xdr_ypresp_key_val(xdrs, &out);

	/*
	 * If we succeeded, call the callback function.
	 * The callback will return TRUE when it wants
	 * no more values.  If we fail, indicate the
	 * error.
	 */
	if (status) {
	    if ((*incallback->foreach)(out.stat,
				       (char *)out.key.keydat_val, out.key.keydat_len,
				       (char *)out.val.valdat_val, out.val.valdat_len,
				       incallback->data))
		return TRUE;
	} else
	    return FALSE;
    }
}
libnsl_hidden_nolink_def (xdr_ypall, GLIBC_2_2)
