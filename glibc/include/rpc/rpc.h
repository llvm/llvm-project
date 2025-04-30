#ifndef _RPC_RPC_H
#include <sunrpc/rpc/rpc.h>

# ifndef _ISOMAC

#include <libc-tsd.h>

/* Now define the internal interfaces.  */
extern unsigned long _create_xid (void);

/*
 * Multi-threaded support
 * Group all global and static variables into a single spot.
 * This area is allocated on a per-thread basis
 */
struct rpc_thread_variables {
	fd_set		svc_fdset_s;		/* Global, rpc_common.c */
	struct rpc_createerr rpc_createerr_s;	/* Global, rpc_common.c */
	struct pollfd	*svc_pollfd_s;		/* Global, rpc_common.c */
	int		svc_max_pollfd_s;	/* Global, rpc_common.c */

	char		*clnt_perr_buf_s;	/* clnt_perr.c */

	struct clntraw_private_s *clntraw_private_s;	/* clnt_raw.c */

	struct callrpc_private_s *callrpc_private_s;	/* clnt_simp.c */

	struct key_call_private *key_call_private_s;	/* key_call.c */

	struct cache_entry *authdes_cache_s;	/* svcauth_des.c */
	int		*authdes_lru_s;		/* svcauth_des.c */

	SVCXPRT		**svc_xports_s;		/* svc.c */
	struct svc_callout *svc_head_s;		/* svc.c */

	struct svcraw_private_s *svcraw_private_s;	/* svc_raw.c */

	struct proglst_ *svcsimple_proglst_s;	/* svc_simple.c */
	SVCXPRT		*svcsimple_transp_s;	/* svc_simple.c */
};

extern struct rpc_thread_variables *__rpc_thread_variables(void)
     __attribute__ ((const)) attribute_hidden;
extern void __rpc_thread_svc_cleanup (void) attribute_hidden;
extern void __rpc_thread_clnt_cleanup (void) attribute_hidden;
extern void __rpc_thread_key_cleanup (void) attribute_hidden;

extern void __rpc_thread_destroy (void) attribute_hidden;

__libc_tsd_define (extern, struct rpc_thread_variables *, RPC_VARS)

#define RPC_THREAD_VARIABLE(x) (__rpc_thread_variables()->x)

libc_hidden_proto (__rpc_thread_svc_max_pollfd)
libc_hidden_proto (__rpc_thread_svc_pollfd)
libc_hidden_proto (__rpc_thread_svc_fdset)
libc_hidden_proto (__rpc_thread_createerr)

/* Perform a host name lookup for NAME and return the first IPv4
   address in *ADDR.  Return 0 on success and -1 on error (and set an
   RPC error).  */
int __libc_rpc_gethostbyname (const char *host, struct sockaddr_in *addr)
  attribute_hidden;

# endif /* !_ISOMAC */
#endif
