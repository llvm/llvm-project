// RUN: %clangxx_asan %s -o %t && %run %t
// RUN: %clangxx_asan %s -o %t -stdlib=libstdc++ -static-libstdc++ && %run %t

#include <stdio.h>
static volatile int zero = 0;
inline void pretend_to_do_something(void *x) {
  __asm__ __volatile__("" : : "r" (x) : "memory");
}

__attribute__((noinline))
void ReallyThrow() {
  fprintf(stderr, "ReallyThrow\n");
  try {
    if (zero == 0)
      throw 42;
    else if (zero == 1)
      throw 1.;
  } catch(double x) {
  }
}

__attribute__((noinline))
void Throw() {
  int a, b, c, d, e;
  pretend_to_do_something(&a);
  pretend_to_do_something(&b);
  pretend_to_do_something(&c);
  pretend_to_do_something(&d);
  pretend_to_do_something(&e);
  fprintf(stderr, "Throw stack = %p\n", &a);
  ReallyThrow();
}

__attribute__((noinline))
void CheckStack() {
  int ar[100];
  pretend_to_do_something(ar);
  for (int i = 0; i < 100; i++)
    ar[i] = i;
  fprintf(stderr, "CheckStack stack = %p, %p\n", ar, ar + 100);
}

int main(int argc, char** argv) {
  try {
    Throw();
  } catch(int a) {
    fprintf(stderr, "a = %d\n", a);
  }
  CheckStack();
}

