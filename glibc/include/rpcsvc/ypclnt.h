#ifndef __RPCSVC_YPCLNT_H__
#include <nis/rpcsvc/ypclnt.h>

# ifndef _ISOMAC

struct ypmaplist;
int yp_maplist (const char *, struct ypmaplist **) __THROW;

libnsl_hidden_proto (ypbinderr_string)
libnsl_hidden_proto (yp_bind)
libnsl_hidden_proto (yp_get_default_domain)
libnsl_hidden_proto (ypprot_err)
libnsl_hidden_proto (yp_master)
libnsl_hidden_proto (yp_update)
libnsl_hidden_proto (yperr_string)
libnsl_hidden_proto (yp_unbind)
libnsl_hidden_proto (yp_order)
libnsl_hidden_proto (yp_first)
libnsl_hidden_proto (yp_next)
libnsl_hidden_proto (yp_match)
libnsl_hidden_proto (yp_all)
libnsl_hidden_proto (__yp_check)
libnsl_hidden_proto (yp_maplist)

# endif /* !_ISOMAC */
#endif
