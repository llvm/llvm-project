/* S390 uses the generic Linux UAPI but defines SA_RESTORER.  */
#define SA_RESTORER	0x04000000

#define SET_SA_RESTORER(kact, act)             \
  (kact)->sa_restorer = (act)->sa_restorer
#define RESET_SA_RESTORER(act, kact)           \
  (act)->sa_restorer = (kact)->sa_restorer

#include <sysdeps/unix/sysv/linux/kernel_sigaction.h>
