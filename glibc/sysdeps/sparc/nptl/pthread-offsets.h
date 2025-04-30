#include <bits/wordsize.h>

#if __WORDSIZE == 64
# define __PTHREAD_MUTEX_KIND_OFFSET		16
#else
# define __PTHREAD_MUTEX_KIND_OFFSET		12
#endif

#if __WORDSIZE == 64
# define __PTHREAD_RWLOCK_FLAGS_OFFSET		48
#else
# define __PTHREAD_RWLOCK_FLAGS_OFFSET		27
#endif
