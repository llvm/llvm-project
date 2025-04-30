#include <stdint.h>

#define STACK_CHK_GUARD \
  ({ uintptr_t x;						\
     asm ("mov %%fs:%c1, %0" : "=r" (x)				\
	  : "i" (offsetof (tcbhead_t, stack_guard))); x; })

#define POINTER_CHK_GUARD \
  ({ uintptr_t x;						\
     asm ("mov %%fs:%c1, %0" : "=r" (x)				\
	  : "i" (offsetof (tcbhead_t, pointer_guard))); x; })
