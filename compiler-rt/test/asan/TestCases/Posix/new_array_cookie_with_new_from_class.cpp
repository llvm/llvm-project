// Test that we do not poison the array cookie if the operator new is defined
// inside the class.
// RUN: %clangxx_asan  %s -o %t && %run %t
//
// XFAIL: target=arm{{.*}}

// UNSUPPORTED: ios

#include <new>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
struct Foo {
  void *operator new(size_t s) { return Allocate(s); }
  void *operator new[] (size_t s) { return Allocate(s); }
  ~Foo();
  static void *allocated;
  static void *Allocate(size_t s) {
    assert(!allocated);
    return allocated = ::new char[s];
  }
};

Foo::~Foo() {}
void *Foo::allocated;

Foo *getFoo(size_t n) {
  return new Foo[n];
}

int main() {
  Foo *foo = getFoo(10);
  fprintf(stderr, "foo  : %p\n", foo);
  fprintf(stderr, "alloc: %p\n", Foo::allocated);
  assert(reinterpret_cast<uintptr_t>(foo) ==
         reinterpret_cast<uintptr_t>(Foo::allocated) + sizeof(void*));
  *reinterpret_cast<uintptr_t*>(Foo::allocated) = 42;
  return 0;
}
