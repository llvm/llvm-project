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
void libcpp_assert_padding(const T& obj, unsigned char pad_byte) {
#ifdef _LIBCPP_VERSION
  assert_padding(obj, pad_byte);
#else
  (void)f;
  (void)pad_byte;
#endif
}

template <class T>
void test() {
  {
    // compare_exchange_strong
    // CAS should succeed when only padding differs in expected; expected is unchanged.
    std::atomic<T> a;

    T init = make<T>(10, 'a', 0xBB);
    assert_padding(init, 0xBB);
    a.store(init);

    T expected = make<T>(10, 'a', 0xAA);
    assert_padding(expected, 0xAA);

    T original_expected; // make a copy including padding bits
    std::memcpy(&original_expected, &expected, sizeof(T));

    T new_value = make<T>(42, 'b', 0xCC);
    assert_padding(new_value, 0xCC);

    bool r = a.compare_exchange_strong(expected, new_value);

    assert(r);
    assert(std::memcmp(&expected, &original_expected, sizeof(T)) == 0);
    T loaded = a.load();
    assert(loaded.i == 42);
    assert(loaded.c == 'b');
    // libc++ always maintains the invariant of the atomic to have zeros in the padding bits
    libcpp_assert_padding(loaded, 0);
  }

  {
    // compare_exchange_strong
    // atomic and expected values are different; failure
    std::atomic<T> a;
    T stored = make<T>(10, 'a', 0xBB);
    assert_padding(stored, 0xBB);
    a.store(stored);

    T expected = make<T>(99, 'a', 0xAA);
    assert_padding(expected, 0xAA);
    T new_value = make<T>(42, 'b', 0xCC);
    assert_padding(new_value, 0xCC);

    bool r = a.compare_exchange_strong(expected, new_value);

    assert(!r);
    assert(expected.i == 10);
    assert(expected.c == 'a');
    // expected is updated to contain atomic's value and in libc++, the paddings bits are always zero
    libcpp_assert_padding(expected, 0);
    T loaded = a.load();
    assert(loaded.i == 10);
    assert(loaded.c == 'a');
    // libc++ always maintains the invariant of the atomic to have zeros in the padding bits
    libcpp_assert_padding(loaded, 0);
  }

  {
    // compare_exchange_weak
    // atomic and expected only differs in padding bits. It should either succeed or spuriously fail
    std::atomic<T> a;
    T stored = make<T>(10, 'a', 0xBB);
    assert_padding(stored, 0xBB);
    a.store(stored);

    T new_value = make<T>(42, 'b', 0xCC);
    assert_padding(new_value, 0xCC);

    T original_expected = make<T>(10, 'a', 0xAA);
    assert_padding(original_expected, 0xAA);

    bool r                  = false;
    const auto max_attempts = 100;
    auto current_attempt    = 0;
    while (!r) {
      ++current_attempt;
      assert(current_attempt < max_attempts && "compare_exchange_weak did not succeed within 3 seconds");
      T expected = make<T>(10, 'a', 0xAA);
      assert_padding(expected, 0xAA);
      r = a.compare_exchange_weak(expected, new_value);
      if (r) {
        assert(std::memcmp(&expected, &original_expected, sizeof(T)) == 0);
      } else {
        // Spurious failure: expected is updated to the current atomic value.
        assert(expected.i == 10);
        assert(expected.c == 'a');
        // expected is updated to contain atomic's value and in libc++, the paddings bits are always zero
        libcpp_assert_padding(expected, 0);
      }
    }

    T loaded = a.load();
    assert(loaded.i == 42);
    assert(loaded.c == 'b');
    // libc++ always maintains the invariant of the atomic to have zeros in the padding bits
    libcpp_assert_padding(loaded, 0);
  }

  {
    // compare_exchange_strong
    // atomic and expected values are different; failure
    std::atomic<T> a;
    T stored = make<T>(10, 'a', 0xBB);
    assert_padding(stored, 0xBB);
    a.store(stored);

    T expected = make<T>(99, 'a', 0xAA);
    assert_padding(expected, 0xAA);
    T new_value = make<T>(42, 'b', 0xCC);
    assert_padding(new_value, 0xCC);

    bool r = a.compare_exchange_weak(expected, new_value);

    assert(!r);
    assert(expected.i == 10);
    assert(expected.c == 'a');
    // expected is updated to contain atomic's value and in libc++, the paddings bits are always zero
    libcpp_assert_padding(expected, 0);
    T loaded = a.load();
    assert(loaded.i == 10);
    assert(loaded.c == 'a');
    // libc++ always maintains the invariant of the atomic to have zeros in the padding bits
    libcpp_assert_padding(loaded, 0);
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
  test<WithTailPadding>();
  test<WithInternalPadding>();
  test<WithInternalAndTailPadding>();
#endif

  return 0;
}
