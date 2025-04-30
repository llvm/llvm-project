/* Architecture-specific adjustments to siginfo_t.  ia64 version.  */
#ifndef _BITS_SIGINFO_ARCH_H

#define __SI_HAVE_SIGSYS 0

#define __SI_SIGFAULT_ADDL			\
  int _si_imm;					\
  unsigned int _si_flags;			\
  unsigned long int _si_isr;

#ifdef __USE_GNU
# define si_imm		_sifields._sigfault._si_imm
# define si_segvflags	_sifields._sigfault._si_flags
# define si_isr		_sifields._sigfault._si_isr
#endif

#endif
