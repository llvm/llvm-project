// REQUIRES: host-supports-jit
// RUN: cat %s | clang-repl -Xcc -Xclang -Xcc -verify | FileCheck %s

// expected-no-diagnostics

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

#if 0                   \
  #error "Can't be!"    \
#endif

