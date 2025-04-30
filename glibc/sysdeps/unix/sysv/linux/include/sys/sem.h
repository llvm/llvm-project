#ifndef _SYS_SEM_H
# include <sysvipc/sys/sem.h>

# ifndef _ISOMAC

__typeof__ (semtimedop) __semtimedop attribute_hidden;

#if __TIMESIZE == 64
# define __semctl64 __semctl
# define __semtimedop64 __semtimedop
#else
# include <struct___timespec64.h>

extern int __semctl64 (int semid, int semnum, int cmd, ...);
libc_hidden_proto (__semctl64);
extern int __semtimedop64 (int semid, struct sembuf *sops, size_t nsops,
			   const struct __timespec64 *tmo);
libc_hidden_proto (__semtimedop64);
#endif

# endif
#endif
