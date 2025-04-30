#include <stdint.h>

#define STACK_CHK_GUARD \
  ({ uintptr_t x; __asm__ ("ear %0,%%a0; sllg %0,%0,32; ear %0,%%a1; lg %0,0x28(%0)" : "=a" (x)); x; })

/* On s390/s390x there is no unique pointer guard, instead we use the
   same value as the stack guard.  */
#define POINTER_CHK_GUARD					\
  ({								\
    uintptr_t x;						\
    __asm__ ("ear %0,%%a0;"					\
	     "sllg %0,%0,32;"					\
	     "ear %0,%%a1;"					\
	     "lg %0,%1(%0)"					\
	     : "=a" (x)						\
	     : "i" (offsetof (tcbhead_t, stack_guard)));	\
    x;								\
  })
