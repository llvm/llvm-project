#ifndef _RPC_AUTH_DES_H

#include <sunrpc/rpc/auth_des.h>

# ifndef _ISOMAC

libc_hidden_proto (getpublickey)
libc_hidden_proto (getsecretkey)
libc_hidden_proto (rtime)

extern bool_t xdr_authdes_cred (XDR *xdrs, struct authdes_cred *cred);
extern bool_t xdr_authdes_verf (XDR *xdrs,
				struct authdes_verf *verf);
struct svc_req;
struct rpc_msg;
extern enum auth_stat _svcauth_des (struct svc_req *rqst,
				    struct rpc_msg *msg);


libc_hidden_proto (authdes_getucred)
libc_hidden_proto (xdr_authdes_cred)
libc_hidden_proto (xdr_authdes_verf)

# endif /* !_ISOMAC */
#endif
