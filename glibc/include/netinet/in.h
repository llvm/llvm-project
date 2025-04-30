#ifndef _NETINET_IN_H

#include <inet/netinet/in.h>

#ifndef _ISOMAC
libc_hidden_proto (bindresvport)
libc_hidden_proto (in6addr_loopback)
extern __typeof (in6addr_loopback) __in6addr_loopback;
libc_hidden_proto (__in6addr_loopback)
libc_hidden_proto (in6addr_any)
extern __typeof (in6addr_any) __in6addr_any;
libc_hidden_proto (__in6addr_any)
#endif

#endif
