//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// XFAIL: !has-64-bit-atomics

// <atomic>
//
// template <class T>
// class atomic;
//
// static constexpr bool is_always_lock_free;

// Ignore diagnostic about vector types changing the ABI on some targets, since
// that is irrelevant for this test.
// ADDITIONAL_COMPILE_FLAGS(gcc-style-warnings): -Wno-psabi

#include <atomic>
#include <cassert>
#include <concepts>
#include <cstddef>

#include "test_macros.h"
#include "atomic_helpers.h"

template <typename T>
void check_always_lock_free(std::atomic<T> const& a) {
  using InfoT = LockFreeStatusInfo<T>;

  constexpr auto is_always_lock_free = std::atomic<T>::is_always_lock_free;
  ASSERT_SAME_TYPE(decltype(is_always_lock_free), bool const);

  // If we know the status of T for sure, validate the exact result of the function.
  if constexpr (InfoT::status_known) {
    constexpr LockFreeStatus known_status = InfoT::value;
    if constexpr (known_status == LockFreeStatus::always) {
      static_assert(is_always_lock_free, "is_always_lock_free is inconsistent with known lock-free status");
      assert(a.is_lock_free() && "is_lock_free() is inconsistent with known lock-free status");
    } else if constexpr (known_status == LockFreeStatus::never) {
      static_assert(!is_always_lock_free, "is_always_lock_free is inconsistent with known lock-free status");
      assert(!a.is_lock_free() && "is_lock_free() is inconsistent with known lock-free status");
    } else {
      assert(a.is_lock_free() || !a.is_lock_free()); // This is kinda dumb, but we might as well call the function once.
    }
  }

  // In all cases, also sanity-check it based on the implication always-lock-free => lock-free.
  if (is_always_lock_free) {
    auto is_lock_free = a.is_lock_free();
    ASSERT_SAME_TYPE(decltype(is_lock_free), bool);
    assert(is_lock_free);
  }
  ASSERT_NOEXCEPT(a.is_lock_free());
}

#define CHECK_ALWAYS_LOCK_FREE(T)                                                                                      \
  do {                                                                                                                 \
    typedef T type;                                                                                                    \
    type obj{};                                                                                                        \
    std::atomic<type> a(obj);                                                                                          \
    check_always_lock_free(a);                                                                                         \
  } while (0)

void test() {
  char c = 'x';
  check_always_lock_free(std::atomic<char>(c));

  int i = 0;
  check_always_lock_free(std::atomic<int>(i));

  float f = 0.f;
  check_always_lock_free(std::atomic<float>(f));

  int* p = &i;
  check_always_lock_free(std::atomic<int*>(p));

  CHECK_ALWAYS_LOCK_FREE(bool);
  CHECK_ALWAYS_LOCK_FREE(char);
  CHECK_ALWAYS_LOCK_FREE(signed char);
  CHECK_ALWAYS_LOCK_FREE(unsigned char);
#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
  CHECK_ALWAYS_LOCK_FREE(char8_t);
#endif
  CHECK_ALWAYS_LOCK_FREE(char16_t);
  CHECK_ALWAYS_LOCK_FREE(char32_t);
  CHECK_ALWAYS_LOCK_FREE(wchar_t);
  CHECK_ALWAYS_LOCK_FREE(short);
  CHECK_ALWAYS_LOCK_FREE(unsigned short);
  CHECK_ALWAYS_LOCK_FREE(int);
  CHECK_ALWAYS_LOCK_FREE(unsigned int);
  CHECK_ALWAYS_LOCK_FREE(long);
  CHECK_ALWAYS_LOCK_FREE(unsigned long);
  CHECK_ALWAYS_LOCK_FREE(long long);
  CHECK_ALWAYS_LOCK_FREE(unsigned long long);
  CHECK_ALWAYS_LOCK_FREE(std::nullptr_t);
  CHECK_ALWAYS_LOCK_FREE(void*);
  CHECK_ALWAYS_LOCK_FREE(float);
  CHECK_ALWAYS_LOCK_FREE(double);
  CHECK_ALWAYS_LOCK_FREE(long double);
#if __has_attribute(vector_size) && defined(_LIBCPP_VERSION)
  CHECK_ALWAYS_LOCK_FREE(int __attribute__((vector_size(1 * sizeof(int)))));
  CHECK_ALWAYS_LOCK_FREE(int __attribute__((vector_size(2 * sizeof(int)))));
  CHECK_ALWAYS_LOCK_FREE(int __attribute__((vector_size(4 * sizeof(int)))));
  CHECK_ALWAYS_LOCK_FREE(int __attribute__((vector_size(16 * sizeof(int)))));
  CHECK_ALWAYS_LOCK_FREE(int __attribute__((vector_size(32 * sizeof(int)))));
  CHECK_ALWAYS_LOCK_FREE(float __attribute__((vector_size(1 * sizeof(float)))));
  CHECK_ALWAYS_LOCK_FREE(float __attribute__((vector_size(2 * sizeof(float)))));
  CHECK_ALWAYS_LOCK_FREE(float __attribute__((vector_size(4 * sizeof(float)))));
  CHECK_ALWAYS_LOCK_FREE(float __attribute__((vector_size(16 * sizeof(float)))));
  CHECK_ALWAYS_LOCK_FREE(float __attribute__((vector_size(32 * sizeof(float)))));
  CHECK_ALWAYS_LOCK_FREE(double __attribute__((vector_size(1 * sizeof(double)))));
  CHECK_ALWAYS_LOCK_FREE(double __attribute__((vector_size(2 * sizeof(double)))));
  CHECK_ALWAYS_LOCK_FREE(double __attribute__((vector_size(4 * sizeof(double)))));
  CHECK_ALWAYS_LOCK_FREE(double __attribute__((vector_size(16 * sizeof(double)))));
  CHECK_ALWAYS_LOCK_FREE(double __attribute__((vector_size(32 * sizeof(double)))));
#endif // __has_attribute(vector_size) && defined(_LIBCPP_VERSION)
  CHECK_ALWAYS_LOCK_FREE(struct Empty{});
  CHECK_ALWAYS_LOCK_FREE(struct OneInt { int i; });
  CHECK_ALWAYS_LOCK_FREE(struct IntArr2 { int i[2]; });
  CHECK_ALWAYS_LOCK_FREE(struct FloatArr3 { float i[3]; });
  CHECK_ALWAYS_LOCK_FREE(struct LLIArr2 { long long int i[2]; });
  CHECK_ALWAYS_LOCK_FREE(struct LLIArr4 { long long int i[4]; });
  CHECK_ALWAYS_LOCK_FREE(struct LLIArr8 { long long int i[8]; });
  CHECK_ALWAYS_LOCK_FREE(struct LLIArr16 { long long int i[16]; });
  CHECK_ALWAYS_LOCK_FREE(struct Padding {
    char c; /* padding */
    long long int i;
  });
  CHECK_ALWAYS_LOCK_FREE(union IntFloat {
    int i;
    float f;
  });
  CHECK_ALWAYS_LOCK_FREE(enum class CharEnumClass : char{foo});

  // C macro and static constexpr must be consistent.
  enum class CharEnumClass : char { foo };
  static_assert(std::atomic<bool>::is_always_lock_free == (2 == ATOMIC_BOOL_LOCK_FREE), "");
  static_assert(std::atomic<char>::is_always_lock_free == (2 == ATOMIC_CHAR_LOCK_FREE), "");
  static_assert(std::atomic<CharEnumClass>::is_always_lock_free == (2 == ATOMIC_CHAR_LOCK_FREE), "");
  static_assert(std::atomic<signed char>::is_always_lock_free == (2 == ATOMIC_CHAR_LOCK_FREE), "");
  static_assert(std::atomic<unsigned char>::is_always_lock_free == (2 == ATOMIC_CHAR_LOCK_FREE), "");
#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
  static_assert(std::atomic<char8_t>::is_always_lock_free == (2 == ATOMIC_CHAR8_T_LOCK_FREE), "");
#endif
  static_assert(std::atomic<char16_t>::is_always_lock_free == (2 == ATOMIC_CHAR16_T_LOCK_FREE), "");
  static_assert(std::atomic<char32_t>::is_always_lock_free == (2 == ATOMIC_CHAR32_T_LOCK_FREE), "");
  static_assert(std::atomic<wchar_t>::is_always_lock_free == (2 == ATOMIC_WCHAR_T_LOCK_FREE), "");
  static_assert(std::atomic<short>::is_always_lock_free == (2 == ATOMIC_SHORT_LOCK_FREE), "");
  static_assert(std::atomic<unsigned short>::is_always_lock_free == (2 == ATOMIC_SHORT_LOCK_FREE), "");
  static_assert(std::atomic<int>::is_always_lock_free == (2 == ATOMIC_INT_LOCK_FREE), "");
  static_assert(std::atomic<unsigned int>::is_always_lock_free == (2 == ATOMIC_INT_LOCK_FREE), "");
  static_assert(std::atomic<long>::is_always_lock_free == (2 == ATOMIC_LONG_LOCK_FREE), "");
  static_assert(std::atomic<unsigned long>::is_always_lock_free == (2 == ATOMIC_LONG_LOCK_FREE), "");
  static_assert(std::atomic<long long>::is_always_lock_free == (2 == ATOMIC_LLONG_LOCK_FREE), "");
  static_assert(std::atomic<unsigned long long>::is_always_lock_free == (2 == ATOMIC_LLONG_LOCK_FREE), "");
  static_assert(std::atomic<void*>::is_always_lock_free == (2 == ATOMIC_POINTER_LOCK_FREE), "");
  static_assert(std::atomic<std::nullptr_t>::is_always_lock_free == (2 == ATOMIC_POINTER_LOCK_FREE), "");

#if TEST_STD_VER >= 20
  static_assert(std::atomic_signed_lock_free::is_always_lock_free, "");
  static_assert(std::atomic_unsigned_lock_free::is_always_lock_free, "");
#endif
}

int main(int, char**) {
  test();
  return 0;
}
