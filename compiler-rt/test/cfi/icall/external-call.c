// RUN: %clang_cfi -lm -o %t1 %s
// RUN: %t1 c 1 2>&1 | FileCheck --check-prefix=CFI %s
// RUN: %t1 s 2 2>&1 | FileCheck --check-prefix=CFI %s

// This test uses jump tables containing PC-relative references to external
// symbols, which the Mach-O object writer does not currently support.
// XFAIL: darwin

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char **argv) {
  // CFI: 1
  fprintf(stderr, "1\n");

  double (*fn)(double);
  if (argv[1][0] == 's')
    fn = sin;
  else
    fn = cos;

  fn(atof(argv[2]));

  // CFI: 2
  fprintf(stderr, "2\n");
}
