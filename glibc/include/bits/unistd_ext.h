#include_next <bits/unistd_ext.h>

#ifndef _ISOMAC
extern int __close_range (unsigned int lowfd, unsigned int highfd, int flags);
libc_hidden_proto (__close_range);
#endif
