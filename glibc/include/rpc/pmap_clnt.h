#ifndef _RPC_PMAP_CLNT_H
#include <sunrpc/rpc/pmap_clnt.h>

# ifndef _ISOMAC

libc_hidden_proto (pmap_getport)
libc_hidden_proto (pmap_set)
libc_hidden_proto (pmap_unset)

/* Defined in pm_getport.c.  */
extern int __get_socket (struct sockaddr_in *saddr) attribute_hidden;
extern u_short __libc_rpc_getport (struct sockaddr_in *address, u_long program,
				   u_long version, u_int protocol,
				   time_t timeout_sec, time_t tottimeout_sec);
libc_hidden_proto (__libc_rpc_getport)

libc_hidden_proto (clnt_broadcast)
libc_hidden_proto (pmap_getmaps)
libc_hidden_proto (pmap_rmtcall)

# endif /* !_ISOMAC */
#endif /* rpc/pmap_clnt.h */
