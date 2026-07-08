//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03

// atomic_init is deprecated
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// atomic<T>::compare_exchange_weak
// atomic<T>::compare_exchange_strong
// libc++ maintains the invariant of the atomic to have zero for all padding bits

#include <atomic>
#include <cassert>
#include <cstring>

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
T make(int i, char c, unsigned char pad_byte) {
  T obj;
  std::memset(&obj, pad_byte, sizeof(T));
  obj.i = i;
  obj.c = c;
  if constexpr (std::is_same_v<T, WithInternalAndTailPadding>) {
    obj.c2 = c;
  }
  return obj;
}

template <class T>
void assert_padding(const T& obj, unsigned char pad_byte) {
  alignas(T) unsigned char buf[sizeof(T)];
  std::memset(buf, pad_byte, sizeof(T));
  T& reference = *reinterpret_cast<T*>(buf);
  reference.i  = obj.i;
  reference.c  = obj.c;
  if constexpr (std::is_same_v<T, WithInternalAndTailPadding>) {
    reference.c2 = obj.c2;
  }
  assert(std::memcmp(&obj, &reference, sizeof(T)) == 0);
}

template <class T>
void test() {
  {
    // atomic();
    std::atomic<T> a;
    T loaded = a.load();
    assert(loaded.i == 0);
    assert(loaded.c == '\0');
    assert_padding(loaded, 0);
  }

  {
    // atomic(T);
    T init = make<T>(10, 'a', 0xBB);
    assert_padding(init, 0xBB);
    std::atomic<T> a(init);
    T loaded = a.load();
    assert(loaded.i == 10);
    assert(loaded.c == 'a');
    assert_padding(loaded, 0);
  }
  {
    // atomic::store
    std::atomic<T> a;
    T value = make<T>(5, 'x', 0xAB);
    assert_padding(value, 0xAB);
    a.store(value);
    T loaded = a.load();
    assert(loaded.i == 5);
    assert(loaded.c == 'x');
    assert_padding(loaded, 0);
  }
  {
    // atomic::exchange
    T initial = make<T>(1, 'a', 0x00);
    assert_padding(initial, 0x00);
    std::atomic<T> a(initial);
    T new_val = make<T>(2, 'b', 0xCD);
    assert_padding(new_val, 0xCD);
    T old = a.exchange(new_val);
    assert(old.i == 1);
    assert(old.c == 'a');
    assert_padding(old, 0);
    T loaded = a.load();
    assert(loaded.i == 2);
    assert(loaded.c == 'b');
    assert_padding(loaded, 0);
  }
  {
    // atomic_init
    std::atomic<T> a;
    T init = make<T>(7, 'z', 0xEF);
    assert_padding(init, 0xEF);
    std::atomic_init(&a, init);
    T loaded = a.load();
    assert(loaded.i == 7);
    assert(loaded.c == 'z');
    assert_padding(loaded, 0);
  }
}

int main(int, char**) {
// TODO(LLVM-23): Switch to XFAIL: clang-22
#if __has_builtin(__builtin_clear_padding)
  test<WithTailPadding>();
  test<WithInternalPadding>();
  test<WithInternalAndTailPadding>();
#endif
  return 0;
}
