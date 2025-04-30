#ifndef __DES_CRYPT_H__
#include <sunrpc/rpc/des_crypt.h>

# ifndef _ISOMAC

/* Now define the internal interfaces.  */

extern int xencrypt (char *secret, char *passwd);
extern int xdecrypt (char *secret, char *passwd);

libc_hidden_proto (des_setparity)
libc_hidden_proto (ecb_crypt)
libc_hidden_proto (cbc_crypt)
libc_hidden_proto (xencrypt)
libc_hidden_proto (xdecrypt)

# endif /* !_ISOMAC */
#endif
