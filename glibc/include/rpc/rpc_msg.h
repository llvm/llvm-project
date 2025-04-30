#ifndef _RPC_MSG_H
#include <sunrpc/rpc/rpc_msg.h>

# ifndef _ISOMAC

libc_hidden_proto (_seterr_reply)

/* Now define the internal interfaces.  */

extern bool_t xdr_rejected_reply (XDR *xdrs, struct rejected_reply *rr);
extern bool_t xdr_accepted_reply (XDR *xdrs, struct accepted_reply *ar);

libc_hidden_proto (xdr_accepted_reply)
libc_hidden_proto (xdr_callhdr)
libc_hidden_proto (xdr_callmsg)
libc_hidden_proto (xdr_rejected_reply)
libc_hidden_proto (xdr_replymsg)

# endif /* !_ISOMAC */
#endif
