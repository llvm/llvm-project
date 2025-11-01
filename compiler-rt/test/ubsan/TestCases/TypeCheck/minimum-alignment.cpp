// RUN: %clangxx %gmlt -fsanitize=alignment %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

// UNSUPPORTED: i386
// UNSUPPORTED: armv7l

// These sanitizers already overload the new operator so won't compile this test
// UNSUPPORTED: ubsan-msan
// UNSUPPORTED: ubsan-tsan

#include <cassert>
#include <cstdlib>

void *operator new(std::size_t count) {
  constexpr const size_t offset = 8;

  // allocate a bit more so we can safely offset it
  void *ptr = std::malloc(count + offset);

  // verify malloc returned 16 bytes aligned mem
  static_assert(__STDCPP_DEFAULT_NEW_ALIGNMENT__ == 16);
  assert(((std::ptrdiff_t)ptr & (__STDCPP_DEFAULT_NEW_ALIGNMENT__ - 1)) == 0);

  return (char *)ptr + offset;
}

struct Foo {
  void *_cookie1, *_cookie2;
};

static_assert(alignof(Foo) == 8);
int main() {
  // CHECK: runtime error: constructor call with pointer from operator new on misaligned address 0x{{.*}} for type 'Foo', which requires target minimum assumed 16 byte alignment
  Foo *f = new Foo;
  return 0;
}
