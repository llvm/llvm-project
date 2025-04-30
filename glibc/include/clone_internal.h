#ifndef _CLONE3_H
#include_next <clone3.h>

extern __typeof (clone3) __clone3;

/* The internal wrapper of clone/clone2 and clone3.  If __clone3 returns
   -1 with ENOSYS, fall back to clone or clone2.  */
extern int __clone_internal (struct clone_args *__cl_args,
			     int (*__func) (void *__arg), void *__arg);

#ifndef _ISOMAC
libc_hidden_proto (__clone3)
libc_hidden_proto (__clone_internal)
#endif

#endif
