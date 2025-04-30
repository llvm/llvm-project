/* SPARC 'struct __new_sigaction' is similar to generic Linux UAPI with
   a sa_restorer field, even though function is passed as an argument
   to rt_sigaction syscall.  */
#define HAS_SA_RESTORER 1
#include <sysdeps/unix/sysv/linux/kernel_sigaction.h>
