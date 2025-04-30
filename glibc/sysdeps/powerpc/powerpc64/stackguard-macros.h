#include <stdint.h>

#define STACK_CHK_GUARD \
  ({ uintptr_t x; asm ("ld %0,-28688(13)" : "=r" (x)); x; })

#define POINTER_CHK_GUARD \
  ({												\
     uintptr_t x;										\
     asm ("ld %0,%1(13)"										\
	  : "=r" (x)										\
	  : "i" (offsetof (tcbhead_t, pointer_guard) - TLS_TCB_OFFSET - sizeof (tcbhead_t))	\
         );											\
     x;												\
   })
