#ifndef __RPCSVC_YP_H__
#include <nis/rpcsvc/yp.h>

# ifndef _ISOMAC

struct ypall_callback;
bool_t xdr_ypall (XDR *, struct ypall_callback *);

libnsl_hidden_proto (xdr_ypstat)
libnsl_hidden_proto (xdr_ypxfrstat)
libnsl_hidden_proto (xdr_domainname)
libnsl_hidden_proto (xdr_mapname)
libnsl_hidden_proto (xdr_peername)
libnsl_hidden_proto (xdr_keydat)
libnsl_hidden_proto (xdr_valdat)
libnsl_hidden_proto (xdr_ypmap_parms)
libnsl_hidden_proto (xdr_ypresp_key_val)
libnsl_hidden_proto (xdr_ypresp_all)
libnsl_hidden_proto (xdr_yppush_status)
libnsl_hidden_proto (xdr_ypbind_resptype)
libnsl_hidden_proto (xdr_ypbind_binding)
libnsl_hidden_proto (xdr_ypreq_nokey)
libnsl_hidden_proto (xdr_ypmaplist)
libnsl_hidden_proto (xdr_ypreq_key)
libnsl_hidden_proto (xdr_ypresp_val)
libnsl_hidden_proto (xdr_ypresp_maplist)
libnsl_hidden_proto (xdr_ypresp_order)
libnsl_hidden_proto (xdr_ypbind_resp)
libnsl_hidden_proto (xdr_ypresp_master)
libnsl_hidden_proto (xdr_ypreq_xfr)
libnsl_hidden_proto (xdr_ypresp_xfr)
libnsl_hidden_proto (xdr_yppushresp_xfr)
libnsl_hidden_proto (xdr_ypbind_setdom)
libnsl_hidden_proto (xdr_ypall)

# endif /* !_ISOMAC */
#endif
