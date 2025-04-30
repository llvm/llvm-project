#ifndef _SYS_RANDOM_H
#include <stdlib/sys/random.h>

# ifndef _ISOMAC

extern ssize_t __getrandom (void *__buffer, size_t __length,
                            unsigned int __flags) __wur;
libc_hidden_proto (__getrandom)

# endif /* !_ISOMAC */
#endif
