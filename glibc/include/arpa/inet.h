#include <inet/arpa/inet.h>

#ifndef _ISOMAC
/* Variant of inet_aton which rejects trailing garbage.  */
extern int __inet_aton_exact (const char *__cp, struct in_addr *__inp);
libc_hidden_proto (__inet_aton_exact)

libc_hidden_proto (inet_ntop)
libc_hidden_proto (inet_pton)
extern __typeof (inet_pton) __inet_pton;
libc_hidden_proto (__inet_pton)
extern __typeof (inet_makeaddr) __inet_makeaddr;
libc_hidden_proto (__inet_makeaddr)
libc_hidden_proto (inet_netof)
extern __typeof (inet_network) __inet_network;
libc_hidden_proto (__inet_network)
#endif
