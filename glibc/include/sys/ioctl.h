#ifndef _SYS_IOCTL_H
#include <misc/sys/ioctl.h>

# ifndef _ISOMAC

/* Now define the internal interfaces.  */
extern int __ioctl (int __fd, unsigned long int __request, ...);
libc_hidden_proto (__ioctl)

# endif /* !_ISOMAC */
#endif
