#ifndef _RPC_SVC_H
#include <sunrpc/rpc/svc.h>

# ifndef _ISOMAC

libc_hidden_proto (xprt_register)
libc_hidden_proto (xprt_unregister)
libc_hidden_proto (svc_register)
libc_hidden_proto (svc_unregister)
libc_hidden_proto (svcerr_auth)
libc_hidden_proto (svcerr_noprog)
libc_hidden_proto (svcerr_progvers)

/* Now define the internal interfaces.  */
extern SVCXPRT *svcfd_create (int fd, u_int sendsize, u_int recvsize);

extern int svcudp_enablecache (SVCXPRT *transp, u_long size);
extern SVCXPRT *svcunixfd_create (int fd, u_int sendsize, u_int recvsize);

libc_hidden_proto (svc_exit)
libc_hidden_proto (svc_getreq)
libc_hidden_proto (svc_getreqset)
libc_hidden_proto (svc_run)
libc_hidden_proto (svc_sendreply)
libc_hidden_proto (svcerr_decode)
libc_hidden_proto (svcerr_noproc)
libc_hidden_proto (svcerr_systemerr)
libc_hidden_proto (svcerr_weakauth)
libc_hidden_proto (svcfd_create)
libc_hidden_proto (svcraw_create)
libc_hidden_proto (svctcp_create)
libc_hidden_proto (svcudp_bufcreate)
libc_hidden_proto (svcudp_create)
libc_hidden_proto (svcudp_enablecache)
libc_hidden_proto (svcunix_create)
libc_hidden_proto (svcunixfd_create)
libc_hidden_proto (svc_getreq_common)
libc_hidden_proto (svc_getreq_poll)

extern void __svc_accept_failed (void) attribute_hidden;
extern void __svc_wait_on_error (void) attribute_hidden;

# endif /* !_ISOMAC */
#endif
