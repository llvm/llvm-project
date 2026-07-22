//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03
// XFAIL: clang-21, apple-clang-21, clang-22

// atomic_init is deprecated
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// atomic<T>::atomic()
// atomic<T>::atomic(T)
// atomic<T>::store(T)
// atomic<T>::exchange(T)
// atomic_init(T)
// libc++ maintains the invariant of the atomic to have zero for all padding bits

#include <atomic>
#include <cassert>
#include <cstring>
#include <type_traits>

#include "test_macros.h"

struct WithTailPadding {
  int i;
  char c;
};

static_assert(sizeof(WithTailPadding) > sizeof(int) + sizeof(char), "");

struct WithInternalPadding {
  char c;
  int i;
};

static_assert(sizeof(WithInternalPadding) > sizeof(int) + sizeof(char), "");

struct WithInternalAndTailPadding {
  char c;
  int i;
  char c2;
};

static_assert(sizeof(WithInternalAndTailPadding) > sizeof(int) + 2 * sizeof(char), "");

template <class T>
void set(T& obj, int i, char c) {
  obj.i = i;
  obj.c = c;
}

void set(WithInternalAndTailPadding& obj, int i, char c) {
  obj.i  = i;
  obj.c  = c;
  obj.c2 = c;
}

template <class T>
void initialize(T& obj, int i, char c, unsigned char pad_byte) {
  std::memset(&obj, pad_byte, sizeof(T));
  set(obj, i, c);
}

template <class T>
void assert_padding(const T& obj, unsigned char pad_byte) {
  alignas(T) unsigned char buf[sizeof(T)];
  std::memset(buf, pad_byte, sizeof(T));
  T& reference = *reinterpret_cast<T*>(buf);
  set(reference, obj.i, obj.c);
  assert(std::memcmp(&obj, &reference, sizeof(T)) == 0);
}

template <class T>
void assert_padding(const std::atomic<T>& obj, unsigned char pad_byte) {
  alignas(T) unsigned char buf[sizeof(T)];
  std::memset(buf, pad_byte, sizeof(T));
  T& reference = *reinterpret_cast<T*>(buf);
  T loaded     = obj.load();
  set(reference, loaded.i, loaded.c);
  assert(std::memcmp(&obj, &reference, sizeof(T)) == 0);
}

template <class T>
void test() {
  {
    // atomic();
#if TEST_STD_VER >= 20
    std::atomic<T> a;
    assert_padding(a, 0);
    T loaded = a.load();
    assert(loaded.i == 0);
    assert(loaded.c == '\0');
#endif
  }

  {
    // atomic(T);
    T init;
    initialize(init, 10, 'a', 0xBB);
    assert_padding(init, 0xBB);
    std::atomic<T> a(init);
    T loaded = a.load();
    assert(loaded.i == 10);
    assert(loaded.c == 'a');
    assert_padding(a, 0);
  }
  {
    // atomic::store
    std::atomic<T> a;
    T value;
    initialize(value, 5, 'x', 0xAB);
    assert_padding(value, 0xAB);
    a.store(value);
    T loaded = a.load();
    assert(loaded.i == 5);
    assert(loaded.c == 'x');
    assert_padding(a, 0);
  }
  {
    // atomic::exchange
    T initial;
    initialize(initial, 1, 'a', 0x00);
    assert_padding(initial, 0x00);
    std::atomic<T> a(initial);
    T new_val;
    initialize(new_val, 2, 'b', 0xCD);
    assert_padding(new_val, 0xCD);
    T old = a.exchange(new_val);
    assert(old.i == 1);
    assert(old.c == 'a');
    T loaded = a.load();
    assert(loaded.i == 2);
    assert(loaded.c == 'b');
    assert_padding(a, 0);
  }
  {
    // atomic_init
    std::atomic<T> a;
    T init;
    initialize(init, 7, 'z', 0xEF);
    assert_padding(init, 0xEF);
    std::atomic_init(&a, init);
    T loaded = a.load();
    assert(loaded.i == 7);
    assert(loaded.c == 'z');
    assert_padding(a, 0);
  }
}

int main(int, char**) {
  test<WithTailPadding>();
  test<WithInternalPadding>();
  test<WithInternalAndTailPadding>();

  return 0;
}
