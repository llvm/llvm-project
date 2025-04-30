#include <stdint.h>

#define STACK_CHK_GUARD \
  ({ uintptr_t x; asm ("ld [%%g7+0x14], %0" : "=r" (x)); x; })

#define POINTER_CHK_GUARD \
  ({ uintptr_t x; asm ("ld [%%g7+0x18], %0" : "=r" (x)); x; })
