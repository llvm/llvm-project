//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: !has-64-bit-atomics
// UNSUPPORTED: !non-lockfree-atomics

#include <cassert>
#include <cstdio>
#include <cstring>
#include <new>
#include <atomic>

template <class T>
void print_bytes(const T* object) {
  auto size                        = sizeof(T);
  const unsigned char* const bytes = reinterpret_cast<const unsigned char*>(object);
  size_t i;

  fprintf(stderr, "[ ");
  for (i = 0; i < size; i++) {
    fprintf(stderr, "%02x ", bytes[i]);
  }
  fprintf(stderr, "]\n");
}

struct Foo {
  int i;
  char c;
};

int main(int, char**) {
  std::atomic<Foo> a;

  Foo init;
  memset(&init, 43, sizeof(Foo));
  init.c = 'a';
  init.i = 10;

  a.store(init);
  print_bytes(&a);

  Foo expected;
  memset(&expected, 42, sizeof(Foo));
  expected.c = 'a';
  expected.i = 10;

  auto r = a.compare_exchange_strong(expected, Foo{42, 'b'});
  assert(r);
}
