// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_umodoi3
// REQUIRES: int256

#include "int_lib.h"
#include <stdio.h>

#ifdef CRT_HAS_256BIT

COMPILER_RT_ABI ou_int __umodoi3(ou_int a, ou_int b);

int test__umodoi3(ou_int a, ou_int b, ou_int expected) {
  ou_int x = __umodoi3(a, b);
  if (x != expected) {
    printf("error in __umodoi3\n");
    return 1;
  }
  return 0;
}

char assumption_1[sizeof(oi_int) == 2 * sizeof(ti_int)] = {0};

#endif

int main() {
#ifdef CRT_HAS_256BIT
  if (test__umodoi3((ou_int)0, (ou_int)1, (ou_int)0))
    return 1;
  if (test__umodoi3((ou_int)10, (ou_int)3, (ou_int)1))
    return 1;
  if (test__umodoi3((ou_int)100, (ou_int)7, (ou_int)2))
    return 1;
  if (test__umodoi3((ou_int)42, (ou_int)42, (ou_int)0))
    return 1;
  if (test__umodoi3((ou_int)3, (ou_int)10, (ou_int)3))
    return 1;
  // (1 << 128) % 2 = 0
  if (test__umodoi3(make_ou(make_tu(0, 1), make_tu(0, 0)), (ou_int)2,
                    (ou_int)0))
    return 1;
  // (1 << 128) % 3 = 1
  if (test__umodoi3(make_ou(make_tu(0, 1), make_tu(0, 0)), (ou_int)3,
                    (ou_int)1))
    return 1;
  // All-ones % 2 = 1
  if (test__umodoi3((ou_int)-1, (ou_int)2, (ou_int)1))
    return 1;
  // Cross-half boundary value mod small
  if (test__umodoi3(make_ou(make_tu(0, 1), make_tu(0, 5)), (ou_int)4,
                    (ou_int)1))
    return 1;
  // Large mod large (same value)
  {
    ou_int big = make_ou(make_tu(0, 0x100), make_tu(0, 0));
    if (test__umodoi3(big, big, (ou_int)0))
      return 1;
  }
  // Large mod large (double)
  {
    ou_int big = make_ou(make_tu(0, 0x100), make_tu(0, 0));
    ou_int dbl = make_ou(make_tu(0, 0x200), make_tu(0, 0));
    if (test__umodoi3(dbl, big, (ou_int)0))
      return 1;
  }
  // Full-width big-number test (all 4 limbs populated).
  // A % B (unsigned), verified by Python: q*b + r == a.
  // Expected value verified by Python arbitrary-precision arithmetic.
  if (test__umodoi3(
          make_ou(make_tu(0xAAAABBBBCCCCDDDDULL, 0xEEEEFFFF11112222ULL),
                  make_tu(0x3333444455556666ULL, 0x7777888899990000ULL)),
          make_ou(make_tu(0x1111222233334444ULL, 0x5555666677778888ULL),
                  make_tu(0x9999AAAABBBBCCCCULL, 0xDDDDEEEEFFFF1111ULL)),
          make_ou(make_tu(0x11108887FFFF7776ULL, 0xEEEE6664DDDD5554ULL),
                  make_tu(0xCCCC4443BBBB3332ULL, 0xAAAA222199A16667ULL))))
    return 1;
#else
  printf("skipped\n");
#endif
  return 0;
}
