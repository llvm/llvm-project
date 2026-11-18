// RUN: %clangxx_cfi -o %t1 %s
// RUN: %run %t1

// RUN: %clangxx_cfi -fsanitize=cfi-cast-strict -o %t2 %s
// RUN: %expect_crash %run %t2

// Without -fsanitize=cfi-cast-strict, CFI allows deleting a base object
// through a pointer to a derived class with the same layout. This must also
// hold with -fwhole-program-vtables (the devirt test suites).

#include "utils.h"

struct A {
  virtual ~A();
};
A::~A() {}

struct B : A {};

int main() {
  A *a = new A;
  break_optimization(a);
  delete static_cast<B *>(a); // UB, allowed without cfi-cast-strict
}
