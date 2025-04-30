#include <time/sys/timeb.h>

#ifndef _ISOMAC
# if __TIMESIZE == 64
#  define __timeb64  timeb
#  define __ftime64  ftime
# else
#  include <struct___timeb64.h>

extern int __ftime64 (struct __timeb64 *) __nonnull ((1));
libc_hidden_proto (__ftime64);
# endif
#endif
