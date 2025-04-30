#include <stdint.h>

#define STACK_CHK_GUARD \
  ({ uintptr_t x; asm ("adds %0 = -8, r13;; ld8 %0 = [%0]" : "=r" (x)); x; })

#define POINTER_CHK_GUARD \
  ({ uintptr_t x; asm ("adds %0 = -16, r13;; ld8 %0 = [%0]" : "=r" (x)); x; })
