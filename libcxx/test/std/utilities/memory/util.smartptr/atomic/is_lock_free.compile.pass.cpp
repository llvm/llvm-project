//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// UNSUPPORTED: no-threads

// [util.smartptr.atomic.shared]
// bool is_lock_free() const noexcept;

// [util.smartptr.atomic.weak]
// bool is_lock_free() const noexcept;

#include <atomic>
#include <memory>

#include "atomic_smart_ptr_test_types.h"
#include "test_macros.h"

template <class T>
void check(const std::atomic<std::shared_ptr<T>>& asp) noexcept {
  static_assert(!std::atomic<std::shared_ptr<T>>::is_always_lock_free);
  std::same_as<bool> decltype(auto) lf = asp.is_lock_free();
  (void)lf;
  ASSERT_SAME_TYPE(decltype(asp.is_lock_free()), bool);
  ASSERT_NOEXCEPT(asp.is_lock_free());
}

template <class T>
void check(const std::atomic<std::weak_ptr<T>>& awp) noexcept {
  static_assert(!std::atomic<std::weak_ptr<T>>::is_always_lock_free);
  std::same_as<bool> decltype(auto) lf = awp.is_lock_free();
  (void)lf;
  ASSERT_SAME_TYPE(decltype(awp.is_lock_free()), bool);
  ASSERT_NOEXCEPT(awp.is_lock_free());
}

int main(int, char**) {
#define LIBCXX_ATOMIC_SP_RUN_IS_LOCK_FREE(T) check<T>(std::atomic<std::shared_ptr<T>>());
  LIBCXX_ATOMIC_SP_FOR_ALL_RUNTIME_TYPES(LIBCXX_ATOMIC_SP_RUN_IS_LOCK_FREE)
#undef LIBCXX_ATOMIC_SP_RUN_IS_LOCK_FREE
#define LIBCXX_ATOMIC_SP_RUN_W_ILF(T) check<T>(std::atomic<std::weak_ptr<T>>());
  LIBCXX_ATOMIC_SP_FOR_ALL_RUNTIME_TYPES(LIBCXX_ATOMIC_SP_RUN_W_ILF)
#undef LIBCXX_ATOMIC_SP_RUN_W_ILF
  return 0;
}
