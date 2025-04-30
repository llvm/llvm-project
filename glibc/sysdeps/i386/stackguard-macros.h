#include <stdint.h>

#define STACK_CHK_GUARD \
  ({ uintptr_t x; asm ("movl %%gs:0x14, %0" : "=r" (x)); x; })

#define POINTER_CHK_GUARD \
  ({							\
     uintptr_t x;					\
     asm ("movl %%gs:%c1, %0" : "=r" (x)		\
	  : "i" (offsetof (tcbhead_t, pointer_guard)));	\
     x;							\
   })
