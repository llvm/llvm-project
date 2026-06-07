//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: !has-64-bit-atomics

// atomic_init is deprecated
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

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

#if __has_builtin(__builtin_clear_padding)

void test_default_constructor() {
  std::atomic<Foo> a;
  Foo loaded = a.load();
  assert(loaded.i == 0);
  assert(loaded.c == '\0');
  assert_foo_padding(loaded, 0);
}

void test_value_constructor() {
  Foo init = make_foo(10, 'a', 0xBB);
  assert_foo_padding(init, 0xBB);
  std::atomic<Foo> a(init);
  Foo loaded = a.load();
  assert(loaded.i == 10);
  assert(loaded.c == 'a');
  assert_foo_padding(loaded, 0);
}

void test_store() {
  std::atomic<Foo> a;
  Foo value = make_foo(5, 'x', 0xAB);
  assert_foo_padding(value, 0xAB);
  a.store(value);
  Foo loaded = a.load();
  assert(loaded.i == 5);
  assert(loaded.c == 'x');
  assert_foo_padding(loaded, 0);
}

void test_exchange() {
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

void test_atomic_init() {
  std::atomic<Foo> a;
  Foo init = make_foo(7, 'z', 0xEF);
  assert_foo_padding(init, 0xEF);
  std::atomic_init(&a, init);
  Foo loaded = a.load();
  assert(loaded.i == 7);
  assert(loaded.c == 'z');
  assert_foo_padding(loaded, 0);
}

void test_compare_exchange_strong_success_padding_only() {
  // CAS should succeed when only padding differs in expected; expected is unchanged.
  std::atomic<Foo> a;

  Foo init = make_foo(10, 'a', 0xBB);
  assert_foo_padding(init, 0xBB);
  a.store(init);

  Foo expected = make_foo(10, 'a', 0xAA);
  assert_foo_padding(expected, 0xAA);

  alignas(Foo) char original_expected[sizeof(Foo)];
  std::memcpy(original_expected, &expected, sizeof(Foo));

  Foo new_value = make_foo(42, 'b', 0xCC);
  assert_foo_padding(new_value, 0xCC);

  bool r = a.compare_exchange_strong(expected, new_value);

  assert(r);
  assert(std::memcmp(&expected, original_expected, sizeof(Foo)) == 0);
  Foo loaded = a.load();
  assert(loaded.i == 42);
  assert(loaded.c == 'b');
  assert_foo_padding(loaded, 0);
}

void test_compare_exchange_strong_failure() {
  std::atomic<Foo> a;
  Foo stored = make_foo(10, 'a', 0xBB);
  assert_foo_padding(stored, 0xBB);
  a.store(stored);

  Foo expected = make_foo(99, 'a', 0xAA);
  assert_foo_padding(expected, 0xAA);
  Foo new_value = make_foo(42, 'b', 0xCC);
  assert_foo_padding(new_value, 0xCC);

  bool r = a.compare_exchange_strong(expected, new_value);

  assert(!r);
  assert(expected.i == 10);
  assert(expected.c == 'a');
  assert_foo_padding(expected, 0);
  Foo loaded = a.load();
  assert(loaded.i == 10);
  assert(loaded.c == 'a');
  assert_foo_padding(loaded, 0);
}

void test_compare_exchange_weak_success_padding_only() {
  std::atomic<Foo> a;
  Foo stored = make_foo(10, 'a', 0xBB);
  assert_foo_padding(stored, 0xBB);
  a.store(stored);

  Foo new_value = make_foo(42, 'b', 0xCC);
  assert_foo_padding(new_value, 0xCC);

  Foo original_expected = make_foo(10, 'a', 0xAA);
  assert_foo_padding(original_expected, 0xAA);

  bool r              = false;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (!r) {
    assert(std::chrono::steady_clock::now() < deadline && "compare_exchange_weak did not succeed within 3 seconds");
    Foo expected = make_foo(10, 'a', 0xAA);
    assert_foo_padding(expected, 0xAA);
    r = a.compare_exchange_weak(expected, new_value);
    if (r) {
      assert(std::memcmp(&expected, &original_expected, sizeof(Foo)) == 0);
    } else {
      // Spurious failure: expected is updated to the current atomic value.
      assert(expected.i == 10);
      assert(expected.c == 'a');
      assert_foo_padding(expected, 0);
    }
  }

  Foo loaded = a.load();
  assert(loaded.i == 42);
  assert(loaded.c == 'b');
  assert_foo_padding(loaded, 0);
}

void test_compare_exchange_weak_failure() {
  std::atomic<Foo> a;
  Foo stored = make_foo(10, 'a', 0xBB);
  assert_foo_padding(stored, 0xBB);
  a.store(stored);

  Foo expected = make_foo(99, 'a', 0xAA);
  assert_foo_padding(expected, 0xAA);
  Foo new_value = make_foo(42, 'b', 0xCC);
  assert_foo_padding(new_value, 0xCC);

  bool r = a.compare_exchange_weak(expected, new_value);

  assert(!r);
  assert(expected.i == 10);
  assert(expected.c == 'a');
  assert_foo_padding(expected, 0);
  Foo loaded = a.load();
  assert(loaded.i == 10);
  assert(loaded.c == 'a');
  assert_foo_padding(loaded, 0);
}

void test_no_padding_type() {
  // Types with unique object representations skip the padding-clearing path.
  std::atomic<int> a(1);
  int expected = 1;
  assert(a.compare_exchange_strong(expected, 2));
  assert(expected == 1);
  assert(a.load() == 2);

  expected = 3;
  assert(!a.compare_exchange_strong(expected, 4));
  assert(expected == 2);
  assert(a.load() == 2);
}

int main(int, char**) {
  test_default_constructor();
  test_value_constructor();
  test_store();
  test_exchange();
  test_atomic_init();
  test_compare_exchange_strong_success_padding_only();
  test_compare_exchange_strong_failure();
  test_compare_exchange_weak_success_padding_only();
  test_compare_exchange_weak_failure();
  test_no_padding_type();

  return 0;
}

#else // !__has_builtin(__builtin_clear_padding)

int main(int, char**) { return 0; }

#endif // __has_builtin(__builtin_clear_padding)
