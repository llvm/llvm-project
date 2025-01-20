// RUN: %clangxx -fsanitize=local-bounds %s -O3 -o %t && %run %t 1
// RUN: %clangxx -fsanitize=local-bounds %s -O3 -o %t && not --crash %run %t 3
// RUN: %clangxx -fsanitize=local-bounds -fno-sanitize-trap=local-bounds %s -O3 -o %t && not %run %t 3 2>&1 | FileCheck %s
// RUN: %clangxx -fsanitize=local-bounds -fno-sanitize-trap=local-bounds -fsanitize-recover=local-bounds %s -O3 -o %t && %run %t 3 2>&1 | FileCheck %s
// RUN: %clangxx -fsanitize=local-bounds -fno-sanitize-trap=local-bounds -fsanitize-recover=local-bounds -g %s -O3 -o %t && %run %t 3 2>&1 | FileCheck %s --check-prefixes=LINE

#include <cstdlib>

struct S {
  int k;
  int l;
};

__attribute__((noinline)) void init(S *s) {
  __asm__ __volatile__("" : : "r"(s) : "memory");
}

__attribute__((noinline, no_sanitize("memory", "address", "hwaddress"))) int
test(char i) {
  S a;
  init(&a);
  S b;
  init(&b);
  return ((int *)(&a))[i];
  // CHECK: error: access out of bounds
  // CHECK: SUMMARY: UndefinedBehaviorSanitizer: undefined-behavior
  // LINE: local_bounds.cpp:[[#@LINE-3]]:{{.*}}runtime error: access out of bounds
  // LINE: SUMMARY: UndefinedBehaviorSanitizer: undefined-behavior {{.*}}local_bounds.cpp:[[#@LINE-4]]
}

int main(int argc, char **argv) {
  test(argv[1][0] - '0');
  return 0;
}
