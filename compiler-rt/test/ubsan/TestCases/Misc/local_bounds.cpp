// RUN: %clangxx -fsanitize=local-bounds %s -O3 -o %t && %run %t 1
// RUN: %clangxx -fsanitize=local-bounds %s -O3 -o %t && not --crash %run %t 3

// FIXME: it's always trap for now.

#include <cstdlib>

struct S {
  int k;
  int l;
};

__attribute__((noinline)) void init(S *s) {
  __asm__ __volatile__("" : : "r"(s) : "memory");
}

__attribute__((noinline, no_sanitize("memory"))) int test(char i) {
  S a;
  init(&a);
  S b;
  init(&b);
  return ((int *)(&a))[i];
}

int main(int argc, char **argv) {
  test(argv[1][0] - '0');
  return 0;
}
