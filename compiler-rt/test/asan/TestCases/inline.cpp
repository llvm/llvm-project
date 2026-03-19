// RUN: %clangxx_asan -O3 %s -o %t && %run %t

// Test that no_sanitize_address attribute applies even when the function would
// be normally inlined.
//
// MSVC doesn't apply __declspec(no_sanitize_address) to inlined functions
//    (i.e. it contains this bug)
// XFAIL: msvc

#include "defines.h"
#include <stdlib.h>

ATTRIBUTE_NO_SANITIZE_ADDRESS
int f(int *p) {
  return *p; // BOOOM?? Nope!
}

int main(int argc, char **argv) {
  int * volatile x = (int*)malloc(2*sizeof(int) + 2);
  int res = f(x + 2);
  free(x);
  if (res)
    exit(0);
  return 0;
}
