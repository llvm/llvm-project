#ifndef _RPC_SVC_AUTH_H
#include <sunrpc/rpc/svc_auth.h>

# ifndef _ISOMAC

/* Now define the internal interfaces.  */
extern enum auth_stat _svcauth_unix (struct svc_req *rqst,
				     struct rpc_msg *msg);
extern enum auth_stat _svcauth_short (struct svc_req *rqst,
				      struct rpc_msg *msg);

libc_hidden_proto (_authenticate)

# endif /* !_ISOMAC */
#endif
