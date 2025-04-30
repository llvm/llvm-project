#include <stdint.h>

#define STACK_CHK_GUARD \
  ({ uintptr_t x; asm ("ldx [%%g7+0x28], %0" : "=r" (x)); x; })

#define POINTER_CHK_GUARD \
  ({ uintptr_t x; asm ("ldx [%%g7+0x30], %0" : "=r" (x)); x; })
