#include <sysdeps/unix/sysv/linux/kernel_sigaction.h>

void __syscall_rt_sigreturn (void) attribute_hidden;
void __syscall_sigreturn (void) attribute_hidden;

#define STUB(act, sigsetsize) \
  (sigsetsize),						\
  (act) ? ((unsigned long)((act->sa_flags & SA_SIGINFO)	\
			    ? &__syscall_rt_sigreturn	\
			    : &__syscall_sigreturn))	\
	: 0
