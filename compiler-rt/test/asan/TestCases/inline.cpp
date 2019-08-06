// RUN: %clangxx_asan -O3 %s -o %t && %run %t

// Test that no_sanitize_address attribute applies even when the function would
// be normally inlined.

#include <stdlib.h>

__attribute__((no_sanitize_address))
int f(int *p) {
  return *p; // BOOOM?? Nope!
}

int main(int argc, char **argv) {
  int * volatile x = (int*)malloc(2*sizeof(int) + 2);
  int res = f(x + 2);
  if (res)
    exit(0);
  return 0;
}
