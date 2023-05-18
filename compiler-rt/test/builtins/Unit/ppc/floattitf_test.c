// REQUIRES: target-is-powerpc64le
// RUN: %clang_builtins %s %librt -o %t && %run %t

/*
 * Test case execution for: long double __floattitf (__int128_t)
 * Conversion from 128 bit integer to long double (IBM double-double).
 */

#include <stdint.h>
#include <stdio.h>

#include "floattitf_test.h"

/* The long double representation, with the high and low portions of
 * the long double, and the corresponding bit patterns of each double. */
typedef union {
  long double ld;
  double d[2]; /* [0] is the high double, [1] is the low double. */
  unsigned long long ull[2]; /* High and low doubles as 64-bit integers. */
} ldUnion;

long double __floattitf(__int128_t);

int main(int argc, char *argv[]) {
  /* Necessary long double and 128 bit integer declarations used to
   * compare computed and expected high and low portions of the
   * IBM double-double. */
  ldUnion expectedLongDouble;
  ldUnion computedLongDouble;
  __int128_t result128;

  for (int i = 0; i < numTests; ++i) {
    /* Set the expected high and low values of the long double,
     * and the 128 bit integer input to be converted. */
    expectedLongDouble.d[0] = tests[i].hi;
    expectedLongDouble.d[1] = tests[i].lo;
    result128 = tests[i].input128;

    /* Get the computed long double from the int128->long double conversion
     * and check for errors between high and low portions. */
    computedLongDouble.ld = __floattitf(result128);

    if ((computedLongDouble.d[0] != expectedLongDouble.d[0]) ||
        (computedLongDouble.d[1] != expectedLongDouble.d[1])) {

      printf("Error on __floattitf( 0x%016llx 0x%016llx ):\n",
             (long long)(tests[i].input128 >> 64),
             (long long)tests[i].input128);
      printf("\tExpected value - %La = ( %a, %a )\n", expectedLongDouble.ld,
             expectedLongDouble.d[0], expectedLongDouble.d[1]);
      printf("\tComputed value - %La = ( %a, %a )\n\n", computedLongDouble.ld,
             computedLongDouble.d[0], computedLongDouble.d[1]);

      return 1;
    }
  }

  return 0;
}
