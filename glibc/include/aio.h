#ifndef _AIO_H
#include <rt/aio.h>

#ifndef _ISOMAC
/* Now define the internal interfaces.  */
extern void __aio_init (const struct aioinit *__init);

/* Flag to signal we need to be compatible with glibc < 2.4 in
   lio_listio and we do not issue events for each individual list
   element.  */
#define LIO_NO_INDIVIDUAL_EVENT	128

# if __TIMESIZE == 64
#  define __aio_suspend_time64 __aio_suspend
# else
extern int __aio_suspend_time64 (const struct aiocb *const list[], int nent,
                                 const struct __timespec64 *timeout);
#  if PTHREAD_IN_LIBC
libc_hidden_proto (__aio_suspend_time64)
#  else
librt_hidden_proto (__aio_suspend_time64)
#endif
# endif
#endif

#endif
