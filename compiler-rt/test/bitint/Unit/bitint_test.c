// RUN: %clang_bitint %s %libbitint -o %t && %run %t

// Ensure that libclang_rt.bitint exports the right symbols

#include "int_lib.h"

#define WORDS 4
int main(int argc, char *argv[]) {
  unsigned int quo[WORDS], rem[WORDS];
  // 'a' needs to have an extra word, see documentation of __udivmodei5.
  unsigned int a[WORDS+1] = {0, 0, 0, 1, 0};
  unsigned int b[WORDS] = {0, 0, 0, 1};

  __udivmodei5(quo, rem, a, b, WORDS);
  __divmodei5(quo, rem, a, b, WORDS);
  return 0;
}
