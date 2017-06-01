// RUN: %clangxx_asan %s -o %t && %run %t
// http://code.google.com/p/address-sanitizer/issues/detail?id=147 (not fixed).
// BROKEN: %clangxx_asan %s -o %t -static-libstdc++ && %run %t
//
// Android builds with static libstdc++ by default.
// XFAIL: android

#include <stdio.h>
static volatile int zero = 0;
inline void pretend_to_do_something(void *x) {
  __asm__ __volatile__("" : : "r" (x) : "memory");
}

__attribute__((noinline, no_sanitize_address))
void ReallyThrow() {
  fprintf(stderr, "ReallyThrow\n");
  if (zero == 0)
    throw 42;
}

__attribute__((noinline))
void Throw() {
  int a, b, c, d, e, f, g, h;
  pretend_to_do_something(&a);
  pretend_to_do_something(&b);
  pretend_to_do_something(&c);
  pretend_to_do_something(&d);
  pretend_to_do_something(&e);
  pretend_to_do_something(&f);
  pretend_to_do_something(&g);
  pretend_to_do_something(&h);
  fprintf(stderr, "Throw stack = %p\n", &a);
  ReallyThrow();
}

__attribute__((noinline))
void CheckStack() {
  int ar[100];
  pretend_to_do_something(ar);
  fprintf(stderr, "CheckStack stack = %p, %p\n", ar, ar + 100);
  for (int i = 0; i < 100; i++)
    ar[i] = i;
}

int main(int argc, char** argv) {
  try {
    Throw();
  } catch(int a) {
    fprintf(stderr, "a = %d\n", a);
  }
  CheckStack();
}
