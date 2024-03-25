// RUN: %clang_cc1 -fsyntax-only -verify -fexceptions %s
// expected-no-diagnostics

#include "Inputs/cuda.h"

__device__ void operator delete(void *p) {}

class A {
  int x;
public:
  A() {
  x = 123;
  }
};

template<class T>
class shared_ptr {
  T *ptr;
public:
  shared_ptr(T *p) {
    ptr = p;
  }
};

// The constructor of B calls the delete operator to clean up
// the memory allocated by the new operator when exceptions happen.
// Make sure that there are no diagnostics due to the device delete
// operator is used.
//
// No need to do similar checks on the device side since it does
// not support exception.
struct B{
  shared_ptr<A> pa{new A};
};

int main() {
  B b;
}
