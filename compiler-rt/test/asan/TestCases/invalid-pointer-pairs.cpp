// RUN: %clangxx_asan -O0 %s -o %t -mllvm -asan-detect-invalid-pointer-pair

// RUN: %env_asan_opts=detect_invalid_pointer_pairs=1 %run %t k 2>&1 | FileCheck %s -check-prefix=OK -allow-empty
// RUN: %env_asan_opts=detect_invalid_pointer_pairs=1 not %run %t g 2>&1 | FileCheck %s -check-prefix=CMP -check-prefix=ALL-ERRORS
// RUN: %env_asan_opts=detect_invalid_pointer_pairs=1 not %run %t s 2>&1 | FileCheck %s -check-prefix=SUB -check-prefix=ALL-ERRORS
// RUN: %env_asan_opts=detect_invalid_pointer_pairs=1 not %run %t f 2>&1 | FileCheck %s -check-prefix=FREE -check-prefix=ALL-ERRORS

#include <assert.h>
#include <stdlib.h>

int f(char c, char *p, char *q) {
  // ALL-ERRORS: ERROR: AddressSanitizer: invalid-pointer-pair
  // [[PTR1:0x[0-9a-f]+]] [[PTR2:0x[0-9a-f]+]]
  switch (c) {
  case 'g':
    // CMP: #{{[0-9]+ .*}} in f({{char, *char *\*, *char *\*}}) {{.*}}invalid-pointer-pairs.cpp:[[@LINE+1]]
    return p > q;
  case 's':
    // SUB: #{{[0-9]+ .*}} in f({{char, *char *\*, *char *\*}}) {{.*}}invalid-pointer-pairs.cpp:[[@LINE+1]]
    return p - q;
  case 'k': {
    // OK-NOT: ERROR
    char *p2 = p + 20;
    return p > p2;
  }
  case 'f': {
    char *p3 = p + 20;
    free(p);
    // FREE: #{{[0-9]+ .*}} in f({{char, *char *\*, *char *\*}}) {{.*}}invalid-pointer-pairs.cpp:[[@LINE+2]]
    // FREE: freed by thread
    return p < p3;
  }
  }
  assert(0);
}

int main(int argc, char **argv) {
  char *p = (char *)malloc(42);
  char *q = (char *)malloc(42);
  assert(argc >= 2);
  f(argv[1][0], p, q);
  free(p);
  free(q);
}
