#ifndef _SYS_SHM_H
#include <sysvipc/sys/shm.h>

#ifndef _ISOMAC

# if __TIMESIZE == 64
#  define __shmctl64 __shmctl
# else
extern int __shmctl64 (int shmid, int cmd, struct __shmid64_ds *buf);
libc_hidden_proto (__shmctl64);
# endif

#endif /* _ISOMAC  */

#endif /* _SYS_SHM_H  */
