// RUN: %clangxx_msan -fno-sanitize-memory-param-retval -O0 %s -o %t && %run %t %p

// PR17377: C++ module destructors get stale argument shadow.

#include <stdio.h>
#include <stdlib.h>
class A {
public:
  // This destructor get stale argument shadow left from the call to f().
  ~A() {
    if (this)
      exit(0);
  }
};

A a;

__attribute__((noinline))
void f(long x) {
}

int main(void) {
  long  x;
  long * volatile p = &x;
  // This call poisons TLS shadow for the first function argument.
  f(*p);
  return 0;
}
