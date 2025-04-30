/* Don't define PTHREAD_STACK_MIN to __sysconf (_SC_THREAD_STACK_MIN)
   for glibc build.  */
#ifdef _ISOMAC
# include <sysdeps/unix/sysv/linux/bits/pthread_stack_min-dynamic.h>
#else
# include <bits/pthread_stack_min.h>
#endif
