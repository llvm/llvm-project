#include <rt/mqueue.h>

#ifndef _ISOMAC
extern __typeof (mq_timedreceive) __mq_timedreceive __nonnull ((2, 5));
extern __typeof (mq_timedsend) __mq_timedsend __nonnull ((2, 5));

# if IS_IN (librt) && !PTHREAD_IN_LIBC
hidden_proto (mq_timedsend)
hidden_proto (__mq_timedsend)
hidden_proto (mq_setattr)
hidden_proto (mq_timedreceive)
hidden_proto (__mq_timedreceive)
# endif

# if PTHREAD_IN_LIBC
libc_hidden_proto (mq_setattr)
libc_hidden_proto (__mq_timedreceive)
libc_hidden_proto (__mq_timedsend)

/* Called from fork so that the new subprocess re-creates the
   notification thread if necessary.  */
void __mq_notify_fork_subprocess (void) attribute_hidden;
# endif

#include <struct___timespec64.h>
#if __TIMESIZE == 64
# define __mq_timedsend_time64 __mq_timedsend
# define __mq_timedreceive_time64 __mq_timedreceive
#else
extern int __mq_timedsend_time64 (mqd_t mqdes, const char *msg_ptr,
                                  size_t msg_len, unsigned int msg_prio,
                                  const struct __timespec64 *abs_timeout);
extern ssize_t __mq_timedreceive_time64 (mqd_t mqdes,
                                         char *__restrict msg_ptr,
                                         size_t msg_len,
                                         unsigned int *__restrict msg_prio,
                                         const struct __timespec64 *__restrict
                                         abs_timeout);
#  if PTHREAD_IN_LIBC
libc_hidden_proto (__mq_timedreceive_time64)
libc_hidden_proto (__mq_timedsend_time64)
#  else
librt_hidden_proto (__mq_timedreceive_time64)
librt_hidden_proto (__mq_timedsend_time64)
#  endif
#endif
#endif
