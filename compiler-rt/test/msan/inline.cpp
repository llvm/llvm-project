// RUN: %clangxx_msan -O3 %s -o %t && %run %t

// Test that no_sanitize_memory attribute applies even when the function would
// be normally inlined.

#include <stdlib.h>

__attribute__((no_sanitize_memory))
int f(int *p) {
  if (*p) // BOOOM?? Nope!
    exit(0);
  return 0;
}

int main(int argc, char **argv) {
  int x;
  int * volatile p = &x;
  int res = f(p);
  return 0;
}
