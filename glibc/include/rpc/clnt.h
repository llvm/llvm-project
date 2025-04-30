#ifndef _RPC_CLNT_H
#include <sunrpc/rpc/clnt.h>

# ifndef _ISOMAC

/* Now define the internal interfaces.  */
extern int _openchild (const char *command, FILE **fto, FILE **ffrom);

extern CLIENT *__libc_clntudp_bufcreate (struct sockaddr_in *__raddr,
					 u_long __program, u_long __version,
					 struct timeval __wait_resend,
					 int *__sockp, u_int __sendsz,
					 u_int __recvsz, int __flags);

libc_hidden_proto (clnt_sperrno)
libc_hidden_proto (clnt_spcreateerror)
libc_hidden_proto (clnt_perror)
libc_hidden_proto (clnt_sperror)
libc_hidden_proto (_rpc_dtablesize)
libc_hidden_proto (callrpc)
libc_hidden_proto (clnt_create)
libc_hidden_proto (clnt_pcreateerror)
libc_hidden_proto (clnt_perrno)
libc_hidden_proto (clntraw_create)
libc_hidden_proto (clnttcp_create)
libc_hidden_proto (clntudp_bufcreate)
libc_hidden_proto (clntudp_create)
libc_hidden_proto (get_myaddress)
libc_hidden_proto (clntunix_create)
libc_hidden_proto (__libc_clntudp_bufcreate)

# endif /* !_ISOMAC */
#endif
