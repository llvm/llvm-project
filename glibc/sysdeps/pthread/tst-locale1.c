/* Test that the thread-local locale works right in the main thread
   when statically linked.  */

#include "../locale/tst-C-locale.c"

#include <pthread.h>
#include <signal.h>

/* This is never called, just here to get pthreads linked in.  */
int
useless (void)
{
  pthread_t th;
  pthread_create (&th, 0, (void *(*) (void *)) useless, 0);
  int result = 0;
#ifdef SIGRTMIN
  /* This is to check __libc_current_sigrt* can be used in statically
     linked apps.  */
  result = SIGRTMIN;
#endif
  return result;
}
