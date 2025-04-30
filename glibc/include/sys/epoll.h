#ifndef _SYS_EPOLL_H
#include_next <sys/epoll.h>

# ifndef _ISOMAC

libc_hidden_proto (epoll_pwait)

# endif /* !_ISOMAC */
#endif
