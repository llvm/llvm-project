//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03
// UNSUPPORTED: no-localization

// atomic<T>::compare_exchange_weak
// atomic<T>::compare_exchange_strong
// CAS should work on types with padding bits

#include <atomic>
#include <cassert>
#include <cstring>
#include <type_traits>
#include <iostream>

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
  std::cerr << "test 1" << std::endl;
  {
    // compare_exchange_strong
    // CAS should succeed when only padding differs in expected; expected is unchanged.
    std::atomic<Foo> a;
    std::cerr << "test 2" << std::endl;

    Foo init = make_foo(10, 'a', 0xBB);
    std::cerr << "test 3" << std::endl;
    assert_foo_padding(init, 0xBB);
    std::cerr << "test 4" << std::endl;
    a.store(init);

    Foo expected = make_foo(10, 'a', 0xAA);
    std::cerr << "test 5" << std::endl;
    assert_foo_padding(expected, 0xAA);
    std::cerr << "test 6" << std::endl;

    Foo original_expected; // make a copy including padding bits
    std::memcpy(&original_expected, &expected, sizeof(Foo));

    Foo new_value = make_foo(42, 'b', 0xCC);
    std::cerr << "test 7" << std::endl;
    assert_foo_padding(new_value, 0xCC);
    std::cerr << "test 8" << std::endl;

    bool r = a.compare_exchange_strong(expected, new_value);

    std::cerr << "test 9" << std::endl;
    assert(r);
    std::cerr << "test 10" << std::endl;
    assert(std::memcmp(&expected, &original_expected, sizeof(Foo)) == 0);
    std::cerr << "test 11" << std::endl;
    Foo loaded = a.load();
    assert(loaded.i == 42);
    std::cerr << "test 12" << std::endl;
    assert(loaded.c == 'b');
    std::cerr << "test 13" << std::endl;
    // libc++ always maintains the invariant of the atomic to have zeros in the padding bits
    libcpp_assert_foo_padding(loaded, 0);
    std::cerr << "test 14" << std::endl;
  }

  {
    // compare_exchange_strong
    // atomic and expected values are different; failure
    std::atomic<Foo> a;
    Foo stored = make_foo(10, 'a', 0xBB);
    std::cerr << "test 15" << std::endl;
    assert_foo_padding(stored, 0xBB);
    std::cerr << "test 16" << std::endl;
    a.store(stored);

    Foo expected = make_foo(99, 'a', 0xAA);
    std::cerr << "test 17" << std::endl;
    assert_foo_padding(expected, 0xAA);
    std::cerr << "test 18" << std::endl;
    Foo new_value = make_foo(42, 'b', 0xCC);
    std::cerr << "test 19" << std::endl;
    assert_foo_padding(new_value, 0xCC);
    std::cerr << "test 10" << std::endl;

    bool r = a.compare_exchange_strong(expected, new_value);

    std::cerr << "test 21" << std::endl;
    assert(!r);
    std::cerr << "test 22" << std::endl;
    assert(expected.i == 10);
    std::cerr << "test 23" << std::endl;
    assert(expected.c == 'a');
    std::cerr << "test 24" << std::endl;
    // expected is updated to contain atomic's value and in libc++, the paddings bits are always zero
    libcpp_assert_foo_padding(expected, 0);
    std::cerr << "test 25" << std::endl;
    Foo loaded = a.load();
    assert(loaded.i == 10);
    std::cerr << "test 26" << std::endl;
    assert(loaded.c == 'a');
    std::cerr << "test 27" << std::endl;
    // libc++ always maintains the invariant of the atomic to have zeros in the padding bits
    libcpp_assert_foo_padding(loaded, 0);
    std::cerr << "test 28" << std::endl;
  }

  {
    // compare_exchange_weak
    // atomic and expected only differs in padding bits. It should either succeed or spuriously fail
    std::atomic<Foo> a;
    Foo stored = make_foo(10, 'a', 0xBB);
    std::cerr << "test 29" << std::endl;
    assert_foo_padding(stored, 0xBB);
    std::cerr << "test 30" << std::endl;
    a.store(stored);

    Foo new_value = make_foo(42, 'b', 0xCC);
    std::cerr << "test 31" << std::endl;
    assert_foo_padding(new_value, 0xCC);
    std::cerr << "test 32" << std::endl;

    Foo original_expected = make_foo(10, 'a', 0xAA);
    std::cerr << "test 33" << std::endl;
    assert_foo_padding(original_expected, 0xAA);
    std::cerr << "test 34" << std::endl;

    bool r                  = false;
    const auto max_attempts = 100;
    auto current_attempt    = 0;
    while (!r) {
      ++current_attempt;
      std::cerr << "test 35" << std::endl;
      assert(current_attempt < max_attempts && "compare_exchange_weak did not succeed within 3 seconds");
      std::cerr << "test 36" << std::endl;
      Foo expected = make_foo(10, 'a', 0xAA);
      assert_foo_padding(expected, 0xAA);
      std::cerr << "test 37" << std::endl;
      r = a.compare_exchange_weak(expected, new_value);
      if (r) {
        std::cerr << "test 38" << std::endl;
        assert(std::memcmp(&expected, &original_expected, sizeof(Foo)) == 0);
        std::cerr << "test 39" << std::endl;
      } else {
        // Spurious failure: expected is updated to the current atomic value.
        std::cerr << "test 40" << std::endl;
        assert(expected.i == 10);
        std::cerr << "test 41" << std::endl;
        assert(expected.c == 'a');
        std::cerr << "test 45" << std::endl;
        // expected is updated to contain atomic's value and in libc++, the paddings bits are always zero
        libcpp_assert_foo_padding(expected, 0);
        std::cerr << "test 43" << std::endl;
      }
    }

    Foo loaded = a.load();
    std::cerr << "test 44" << std::endl;
    assert(loaded.i == 42);
    std::cerr << "test 45" << std::endl;
    assert(loaded.c == 'b');
    std::cerr << "test 46" << std::endl;
    // libc++ always maintains the invariant of the atomic to have zeros in the padding bits
    libcpp_assert_foo_padding(loaded, 0);
    std::cerr << "test 47" << std::endl;
  }

  {
    // compare_exchange_strong
    // atomic and expected values are different; failure
    std::atomic<Foo> a;
    Foo stored = make_foo(10, 'a', 0xBB);
    std::cerr << "test 48" << std::endl;
    assert_foo_padding(stored, 0xBB);
    std::cerr << "test 49" << std::endl;
    a.store(stored);

    Foo expected = make_foo(99, 'a', 0xAA);
    std::cerr << "test 50" << std::endl;
    assert_foo_padding(expected, 0xAA);
    std::cerr << "test 51" << std::endl;
    Foo new_value = make_foo(42, 'b', 0xCC);
    std::cerr << "test 52" << std::endl;
    assert_foo_padding(new_value, 0xCC);
    std::cerr << "test 53" << std::endl;

    bool r = a.compare_exchange_weak(expected, new_value);

    std::cerr << "test 54" << std::endl;
    assert(!r);
    std::cerr << "test 55" << std::endl;
    assert(expected.i == 10);
    std::cerr << "test 56" << std::endl;
    assert(expected.c == 'a');
    std::cerr << "test 57" << std::endl;
    // expected is updated to contain atomic's value and in libc++, the paddings bits are always zero
    libcpp_assert_foo_padding(expected, 0);
    std::cerr << "test 58" << std::endl;
    Foo loaded = a.load();
    std::cerr << "test 59" << std::endl;
    assert(loaded.i == 10);
    std::cerr << "test 60" << std::endl;
    assert(loaded.c == 'a');
    std::cerr << "test 61" << std::endl;
    // libc++ always maintains the invariant of the atomic to have zeros in the padding bits
    libcpp_assert_foo_padding(loaded, 0);
    std::cerr << "test 62" << std::endl;
  }

  {
    // Types with unique object representations skip the padding-clearing path.
    std::atomic<int> a(1);
    int expected = 1;
    std::cerr << "test 63" << std::endl;
    assert(a.compare_exchange_strong(expected, 2));
    std::cerr << "test 64" << std::endl;
    assert(expected == 1);
    std::cerr << "test 65" << std::endl;
    assert(a.load() == 2);
    std::cerr << "test 66" << std::endl;

    expected = 3;
    assert(!a.compare_exchange_strong(expected, 4));
    std::cerr << "test 67" << std::endl;
    assert(expected == 2);
    std::cerr << "test 68" << std::endl;
    assert(a.load() == 2);
    std::cerr << "test 69" << std::endl;
  }
}

int main(int, char**) {
  // TODO(LLVM-23): Switch to XFAIL: clang-22
  std::cerr << "main 1" << std::endl;
#if __has_builtin(__builtin_clear_padding)
  std::cerr << "main 2" << std::endl;
  test();
  std::cerr << "main 3" << std::endl;
#endif
  std::cerr << "main 4" << std::endl;
  return 0;
}
