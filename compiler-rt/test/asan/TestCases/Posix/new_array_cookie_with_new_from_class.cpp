// Test that we do not poison the array cookie if the operator new is defined
// inside the class.
// RUN: %clangxx_asan  %s -o %t && %run %t

#include <new>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>

#if defined(__arm__) || (defined(__aarch64__) && defined(__APPLE__))
constexpr size_t kArrayCookieSize = 2 * sizeof(void *);
#else
constexpr size_t kArrayCookieSize = sizeof(void *);
#endif

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
         reinterpret_cast<uintptr_t>(Foo::allocated) + kArrayCookieSize);
  *reinterpret_cast<uintptr_t*>(Foo::allocated) = 42;
  return 0;
}
