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
// CAS should work on types with padding bits

#include <atomic>
#include <cassert>
#include <cstring>
#include <type_traits>

struct Foo {
  int i;
  char c;
};

static_assert(sizeof(Foo) > sizeof(int) + sizeof(char), "");

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

void libcpp_assert_foo_padding(const Foo& f, unsigned char pad_byte) {
#ifdef _LIBCPP_VERSION
  assert_foo_padding(f, pad_byte);
#else
  (void)f;
  (void)pad_byte;
#endif
}

void test() {
  {
    // compare_exchange_strong
    // CAS should succeed when only padding differs in expected; expected is unchanged.
    std::atomic<Foo> a;

    Foo init = make_foo(10, 'a', 0xBB);
    assert_foo_padding(init, 0xBB);
    a.store(init);

    Foo expected = make_foo(10, 'a', 0xAA);
    assert_foo_padding(expected, 0xAA);

    Foo original_expected; // make a copy including padding bits
    std::memcpy(&original_expected, &expected, sizeof(Foo));

    Foo new_value = make_foo(42, 'b', 0xCC);
    assert_foo_padding(new_value, 0xCC);

    bool r = a.compare_exchange_strong(expected, new_value);

    assert(r);
    assert(std::memcmp(&expected, &original_expected, sizeof(Foo)) == 0);
    Foo loaded = a.load();
    assert(loaded.i == 42);
    assert(loaded.c == 'b');
    // libc++ always maintains the invariant of the atomic to have zeros in the padding bits
    libcpp_assert_foo_padding(loaded, 0);
  }

  {
    // compare_exchange_strong
    // atomic and expected values are different; failure
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
    // expected is updated to contain atomic's value and in libc++, the paddings bits are always zero
    libcpp_assert_foo_padding(expected, 0);
    Foo loaded = a.load();
    assert(loaded.i == 10);
    assert(loaded.c == 'a');
    // libc++ always maintains the invariant of the atomic to have zeros in the padding bits
    libcpp_assert_foo_padding(loaded, 0);
  }

  {
    // compare_exchange_weak
    // atomic and expected only differs in padding bits. It should either succeed or spuriously fail
    std::atomic<Foo> a;
    Foo stored = make_foo(10, 'a', 0xBB);
    assert_foo_padding(stored, 0xBB);
    a.store(stored);

    Foo new_value = make_foo(42, 'b', 0xCC);
    assert_foo_padding(new_value, 0xCC);

    Foo original_expected = make_foo(10, 'a', 0xAA);
    assert_foo_padding(original_expected, 0xAA);

    bool r                  = false;
    const auto max_attempts = 100;
    auto current_attempt    = 0;
    while (!r) {
      ++current_attempt;
      assert(current_attempt < max_attempts && "compare_exchange_weak did not succeed within 3 seconds");
      Foo expected = make_foo(10, 'a', 0xAA);
      assert_foo_padding(expected, 0xAA);
      r = a.compare_exchange_weak(expected, new_value);
      if (r) {
        assert(std::memcmp(&expected, &original_expected, sizeof(Foo)) == 0);
      } else {
        // Spurious failure: expected is updated to the current atomic value.
        assert(expected.i == 10);
        assert(expected.c == 'a');
        // expected is updated to contain atomic's value and in libc++, the paddings bits are always zero
        libcpp_assert_foo_padding(expected, 0);
      }
    }

    Foo loaded = a.load();
    assert(loaded.i == 42);
    assert(loaded.c == 'b');
    // libc++ always maintains the invariant of the atomic to have zeros in the padding bits
    libcpp_assert_foo_padding(loaded, 0);
  }

  {
    // compare_exchange_strong
    // atomic and expected values are different; failure
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
    // expected is updated to contain atomic's value and in libc++, the paddings bits are always zero
    libcpp_assert_foo_padding(expected, 0);
    Foo loaded = a.load();
    assert(loaded.i == 10);
    assert(loaded.c == 'a');
    // libc++ always maintains the invariant of the atomic to have zeros in the padding bits
    libcpp_assert_foo_padding(loaded, 0);
  }

  {
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
}

int main(int, char**) {
// TODO(LLVM-23): Switch to XFAIL: clang-22
#if __has_builtin(__builtin_clear_padding)
  test();
#endif // __has_builtin(__builtin_clear_padding)

  return 0;
}
