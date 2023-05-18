// REQUIRES: host-supports-jit
// UNSUPPORTED: system-aix
// RUN: cat %s | clang-repl | FileCheck %s

extern "C" int printf(const char*,...);
int i = \
  12;

printf("i=%d\n", i);
// CHECK: i=12

void f(int x) \ 
{                                               \
  printf("x=\
          %d", x); \
}
f(i);
// CHECK: x=12

// FIXME: Support preprocessor directives.
// #if 0 \
//   #error "Can't be!" \
// #endif

