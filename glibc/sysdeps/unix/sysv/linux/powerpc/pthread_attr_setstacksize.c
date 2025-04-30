#include <unistd.h>	/* For __getpagesize.  */
#define NEW_VERNUM GLIBC_2_6
#define STACKSIZE_ADJUST \
  do {									      \
    size_t ps = __getpagesize ();					      \
    if (stacksize < 2 * ps)						      \
      stacksize = 2 * ps;						      \
  } while (0)
#include <nptl/pthread_attr_setstacksize.c>
