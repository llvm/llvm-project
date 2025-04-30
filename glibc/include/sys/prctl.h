#ifndef _SYS_PRCTL_H
#include_next <sys/prctl.h>

# ifndef _ISOMAC

extern int __prctl (int __option, ...);
libc_hidden_proto (__prctl)

# endif /* !_ISOMAC */
#endif
