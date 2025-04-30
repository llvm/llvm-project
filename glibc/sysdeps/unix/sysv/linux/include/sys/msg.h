#ifndef _SYS_MSG_H
#include <sysvipc/sys/msg.h>

#ifndef _ISOMAC
extern ssize_t __libc_msgrcv (int msqid, void *msgp, size_t msgsz,
			      long int msgtyp, int msgflg);
extern int __libc_msgsnd (int msqid, const void *msgp, size_t msgsz,
			  int msgflg);

# if __TIMESIZE == 64
#  define __msgctl64 __msgctl
# else
extern int __msgctl64 (int msqid, int cmd, struct __msqid64_ds *buf);
libc_hidden_proto (__msgctl64);
# endif

#endif

#endif
