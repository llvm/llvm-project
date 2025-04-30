#if __WORDSIZE == 64
# define __PTHREAD_MUTEX_KIND_OFFSET		16
#else
# define __PTHREAD_MUTEX_KIND_OFFSET		12
#endif


#if __WORDSIZE == 64
# define __PTHREAD_RWLOCK_FLAGS_OFFSET		48
#else
# if __BYTE_ORDER == __BIG_ENDIAN
#  define __PTHREAD_RWLOCK_FLAGS_OFFSET		27
# else
#  define __PTHREAD_RWLOCK_FLAGS_OFFSET		24
# endif
#endif
