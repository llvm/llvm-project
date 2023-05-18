// REQUIRES: target-is-powerpc64le
// RUN: %clang_builtins %s %librt -o %t && %run %t

#include <stdint.h>
#include <stdio.h>

#include "fixtfti_test.h"

// The long double representation, with the high and low portions of
// the long double, and the corresponding bit patterns of each double.
typedef union {
  long double ld;
  double d[2]; // [0] is the high double, [1] is the low double.
  unsigned long long ull[2]; // High and low doubles as 64-bit integers.
} ldUnion;

__int128_t __fixtfti(long double);

int main(int argc, char *argv[]) {
  // Necessary long double and (unsigned) 128 bit integer
  // declarations used to compare the computed and expected results
  // from converting the IBM double-double to int128.
  ldUnion ldInput;
  __uint128_t expectedResult, computedResult;

  for (int i = 0; i < numTests; ++i) {
    // Set the expected 128 bit integer and the high and low
    // values of the long double input.
    ldInput.d[0] = testList[i].hiInput;
    ldInput.d[1] = testList[i].loInput;
    expectedResult = testList[i].result128;

    // Get the computed 128 bit integer from the long double to
    // int128 conversion. Here we cast to unsigned int128 to
    // get the bit pattern of the result.
    computedResult = (__uint128_t) __fixtfti(ldInput.ld);

    if (computedResult != expectedResult) {
      printf("Error for __fixtfti at input %La = ( %a , %a ):\n", ldInput.ld,
             ldInput.d[0], ldInput.d[1]);
      printf("\tExpected __int128_t: 0x%016llx 0x%016llx\n",
             (unsigned long long)(expectedResult >> 64),
             (unsigned long long)expectedResult);
      printf("\tComputed __int128_t: 0x%016llx 0x%016llx\n\n",
             (unsigned long long)(computedResult >> 64),
             (unsigned long long)computedResult);

      return 1;
    }
  }

  return 0;
}
