#ifndef	_SEMAPHORE_H
#include_next <semaphore.h>

#ifndef _ISOMAC
extern __typeof (sem_post) __sem_post;
libpthread_hidden_proto (__sem_post)
#endif

#endif
