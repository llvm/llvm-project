//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <atomic>

// static constexpr bool is_always_lock_free;
// bool is_lock_free() const noexcept;

#include <atomic>
#include <cassert>
#include <concepts>

#include "test_macros.h"

template <typename T>
void check_always_lock_free(std::atomic_ref<T> const a) {
  std::same_as<const bool> decltype(auto) is_always_lock_free = std::atomic_ref<T>::is_always_lock_free;
  if (is_always_lock_free) {
    std::same_as<bool> decltype(auto) is_lock_free = a.is_lock_free();
    assert(is_lock_free);
  }
  ASSERT_NOEXCEPT(a.is_lock_free());
}

#define CHECK_ALWAYS_LOCK_FREE(T)                                                                                      \
  do {                                                                                                                 \
    typedef T type;                                                                                                    \
    type obj{};                                                                                                        \
    check_always_lock_free(std::atomic_ref<type>(obj));                                                                \
  } while (0)

void test() {
  int i = 0;
  check_always_lock_free(std::atomic_ref<int>(i));

  float f = 0.f;
  check_always_lock_free(std::atomic_ref<float>(f));

  int* p = &i;
  check_always_lock_free(std::atomic_ref<int*>(p));

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
}

int main(int, char**) {
  test();
  return 0;
}
