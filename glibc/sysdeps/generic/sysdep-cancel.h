#include <sysdep.h>

/* No multi-thread handling enabled.  */
#define SINGLE_THREAD_P (1)
#define RTLD_SINGLE_THREAD_P (1)
#define LIBC_CANCEL_ASYNC()	0 /* Just a dummy value.  */
#define LIBC_CANCEL_RESET(val)	((void)(val)) /* Nothing, but evaluate it.  */
