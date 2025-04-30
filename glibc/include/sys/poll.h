#ifndef	_SYS_POLL_H
# include <io/sys/poll.h>

#ifndef _ISOMAC
extern int __poll (struct pollfd *__fds, unsigned long int __nfds,
		   int __timeout);
libc_hidden_proto (__poll)
libc_hidden_proto (ppoll)

# if __TIMESIZE == 64
#  define __ppoll64 __ppoll
# else
# include <time.h>
# include <signal.h>

extern int __ppoll64 (struct pollfd *fds, nfds_t nfds,
                      const struct __timespec64 *timeout,
                      const sigset_t *sigmask);
libc_hidden_proto (__ppoll64)
# endif
#endif
#endif
