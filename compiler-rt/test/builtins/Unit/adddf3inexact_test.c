// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_adddf3

#include "int_lib.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(__arm__) && (__ARM_FP & 0xc) == 4

/* Run test on arm hardware with 32-bit only FPU */
#  define PERFORM_TEST

#  define ARM_INVALID 0x0001
#  define ARM_DIVBYZERO 0x0002
#  define ARM_OVERFLOW 0x0004
#  define ARM_UNDERFLOW 0x0008
#  define ARM_INEXACT 0x0010
#  define ARM_DENORMAL 0x0080
#  define ARM_ALL_EXCEPT                                                       \
    (ARM_DIVBYZERO | ARM_INEXACT | ARM_INVALID | ARM_OVERFLOW |                \
     ARM_UNDERFLOW | ARM_DENORMAL)

static void clear_except(void) {
  uint32_t fpscr;

  /* Clear all exception bits */
  __asm__ __volatile__("vmrs  %0, fpscr" : "=r"(fpscr));
  fpscr &= ~(ARM_ALL_EXCEPT);
  __asm__ __volatile__("vmsr  fpscr, %0" : : "ri"(fpscr));
}

static int get_except(void) {
  uint32_t fpscr;

  /* Test exception bits */
  __asm__ __volatile__("vmrs  %0, fpscr" : "=r"(fpscr));
  return (int)(fpscr & ARM_ALL_EXCEPT);
}

#endif

#ifdef PERFORM_TEST
int test__adddf3inexact(volatile double a, volatile double b,
                        volatile double expected) {
  volatile double actual;
  int ret = 0;
  int except;

  clear_except();
  actual = a + b;
  if (actual != expected && isnan(actual) != isnan(expected)) {
    printf("error in test__adddf3inexact(%a, %a) = %a, expected %a\n", a, b,
           actual, expected);
    ret = 1;
  }

  except = get_except();
  if (except != 0) {
    printf("error in test__adddf3inexact(%a, %a). raised exceptions 0x%x\n", a,
           b, except);
    ret = 1;
  }

  return ret;
}
#endif

int main() {
#ifdef PERFORM_TEST
  if (test__adddf3inexact(0x1p+0, 0x1p-53, 0x1p+0))
    return 1;
  if (test__adddf3inexact(0x1p+0, 0x1p-52, 0x1.0000000000001p+0))
    return 1;
  if (test__adddf3inexact(0x1p+0, NAN, NAN))
    return 1;
#else
  printf("skipped\n");
#endif
  return 0;
}
