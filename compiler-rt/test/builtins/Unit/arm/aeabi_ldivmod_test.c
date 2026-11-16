// REQUIRES: arm-target-arch || armv4t-target-arch
// RUN: %clang_builtins -mthumb %s %librt -o %t && %run %t

#include "int_lib.h"

COMPILER_RT_ABI void __aeabi_ldivmod(di_int a, di_int b);

int test_aeabi_ldivmod(di_int a, di_int b, di_int expected_q,
                       di_int expected_r) {
  di_int q, r;
  // __aeabi_ldivmod returns a struct { quotient; remainder; } using
  // value_in_regs calling convention. Each field is a 64-bit integer, so the
  // quotient resides in r0 and r1, while the remainder in r2 and r3. The
  // byte order however depends on the endianness.
  __asm__(
#if _YUGA_BIG_ENDIAN
      "movs r1, %Q[a] \n"
      "movs r0, %R[a] \n"
      "movs r3, %Q[b] \n"
      "movs r2, %R[b] \n"
#else
      "movs r0, %Q[a] \n"
      "movs r1, %R[a] \n"
      "movs r2, %Q[b] \n"
      "movs r3, %R[b] \n"
#endif
      "bl __aeabi_ldivmod \n"
#if _YUGA_BIG_ENDIAN
      "movs %Q[q], r1 \n"
      "movs %R[q], r0 \n"
      "movs %Q[r], r3 \n"
      "movs %R[r], r2 \n"
#else
      "movs %Q[q], r0 \n"
      "movs %R[q], r1 \n"
      "movs %Q[r], r2 \n"
      "movs %R[r], r3 \n"
#endif
      : [q] "=r"(q), [r] "=r"(r)
      : [a] "r"(a), [b] "r"(b)
      : "lr", "r0", "r1", "r2", "r3");
  return q != expected_q || r != expected_r;
}

int main(void) {
  if (test_aeabi_ldivmod(0, 1, 0, 0))
    return 1;
  if (test_aeabi_ldivmod(19, 5, 3, 4))
    return 1;
  if (test_aeabi_ldivmod(-19, 5, -3, -4))
    return 1;
  if (test_aeabi_ldivmod(0x123456789abcdefLL, -0x1234567LL, -0x100000079LL,
                         0x40LL))
    return 1;
  return 0;
}
