/* m68k does not define SA_RESTORER, but does have sa_restorer member
   on kernel sigaction struct.  */
#define HAS_SA_RESTORER 1
#include <sysdeps/unix/sysv/linux/kernel_sigaction.h>
