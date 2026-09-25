// REQUIRES: arm-target-arch || armv4t-target-arch
// RUN: %clang_builtins -mthumb %s %librt -o %t && %run %t

#include <math.h>

// ARMv4T Thumb code can not read status register. These helpers call
// the EABI comparison functions and turn Z/C flags into numbers:
// -1 for less, 0 for equal, and 1 for greater or unordered
__asm__(".globl call_aeabi_cfcmple\n"
        ".thumb_func\n"
        "call_aeabi_cfcmple:\n"
        "  push {r4, lr}\n"
        "  bl __aeabi_cfcmple\n"
        "  beq 1f\n"
        "  bcs 2f\n"
        "  movs r0, #1\n"
        "  negs r0, r0\n"
        "  b 3f\n"
        "1:\n"
        "  movs r0, #0\n"
        "  b 3f\n"
        "2:\n"
        "  movs r0, #1\n"
        "3:\n"
        "  pop {r4}\n"
        "  pop {r3}\n"
        "  bx r3\n"

        ".globl call_aeabi_cdcmple\n"
        ".thumb_func\n"
        "call_aeabi_cdcmple:\n"
        "  push {r4, lr}\n"
        "  bl __aeabi_cdcmple\n"
        "  beq 1f\n"
        "  bcs 2f\n"
        "  movs r0, #1\n"
        "  negs r0, r0\n"
        "  b 3f\n"
        "1:\n"
        "  movs r0, #0\n"
        "  b 3f\n"
        "2:\n"
        "  movs r0, #1\n"
        "3:\n"
        "  pop {r4}\n"
        "  pop {r3}\n"
        "  bx r3\n");

extern int call_aeabi_cfcmple(float, float);
extern int call_aeabi_cdcmple(double, double);

int main(void) {
  if (call_aeabi_cfcmple(1.0f, 2.0f) != -1)
    return 1;
  if (call_aeabi_cfcmple(1.0f, 1.0f) != 0)
    return 1;
  if (call_aeabi_cfcmple(2.0f, 1.0f) != 1)
    return 1;
  if (call_aeabi_cfcmple(NAN, 1.0f) != 1)
    return 1;

  if (call_aeabi_cdcmple(1.0, 2.0) != -1)
    return 1;
  if (call_aeabi_cdcmple(1.0, 1.0) != 0)
    return 1;
  if (call_aeabi_cdcmple(2.0, 1.0) != 1)
    return 1;
  if (call_aeabi_cdcmple(NAN, 1.0) != 1)
    return 1;
  return 0;
}
