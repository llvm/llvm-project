//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// atomic_init is deprecated
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// atomic<T>::compare_exchange_weak
// atomic<T>::compare_exchange_strong
// libc++ maintains the invariant of the atomic to have zero for all padding bits

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstring>
#include <type_traits>

struct Foo {
  int i;
  char c;
};

static_assert(!std::has_unique_object_representations_v<Foo>);
static_assert(sizeof(Foo) > sizeof(int) + sizeof(char));

Foo make_foo(int i, char c, unsigned char pad_byte) {
  Foo f;
  std::memset(&f, pad_byte, sizeof(Foo));
  f.i = i;
  f.c = c;
  return f;
}

void assert_foo_padding(const Foo& f, unsigned char pad_byte) {
  alignas(Foo) unsigned char buf[sizeof(Foo)];
  std::memset(buf, pad_byte, sizeof(Foo));
  Foo& reference = *reinterpret_cast<Foo*>(buf);
  reference.i    = f.i;
  reference.c    = f.c;
  assert(std::memcmp(&f, &reference, sizeof(Foo)) == 0);
}

void test() {
  {
    // atomic();
    std::atomic<Foo> a;
    Foo loaded = a.load();
    assert(loaded.i == 0);
    assert(loaded.c == '\0');
    assert_foo_padding(loaded, 0);
  }

  {
    // atomic(T);
    Foo init = make_foo(10, 'a', 0xBB);
    assert_foo_padding(init, 0xBB);
    std::atomic<Foo> a(init);
    Foo loaded = a.load();
    assert(loaded.i == 10);
    assert(loaded.c == 'a');
    assert_foo_padding(loaded, 0);
  }
  {
    // atomic::store
    std::atomic<Foo> a;
    Foo value = make_foo(5, 'x', 0xAB);
    assert_foo_padding(value, 0xAB);
    a.store(value);
    Foo loaded = a.load();
    assert(loaded.i == 5);
    assert(loaded.c == 'x');
    assert_foo_padding(loaded, 0);
  }
  {
    // atomic::exchange
    Foo initial = make_foo(1, 'a', 0x00);
    assert_foo_padding(initial, 0x00);
    std::atomic<Foo> a(initial);
    Foo new_val = make_foo(2, 'b', 0xCD);
    assert_foo_padding(new_val, 0xCD);
    Foo old = a.exchange(new_val);
    assert(old.i == 1);
    assert(old.c == 'a');
    assert_foo_padding(old, 0);
    Foo loaded = a.load();
    assert(loaded.i == 2);
    assert(loaded.c == 'b');
    assert_foo_padding(loaded, 0);
  }
  {
    // atomic_init
    std::atomic<Foo> a;
    Foo init = make_foo(7, 'z', 0xEF);
    assert_foo_padding(init, 0xEF);
    std::atomic_init(&a, init);
    Foo loaded = a.load();
    assert(loaded.i == 7);
    assert(loaded.c == 'z');
    assert_foo_padding(loaded, 0);
  }
}

int main(int, char**) {
// TODO(LLVM-23): Switch to XFAIL: clang-22
#if __has_builtin(__builtin_clear_padding)
  test();
#endif
  return 0;
}
