//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// UNSUPPORTED: no-threads

// static constexpr bool is_always_lock_free;
// bool is_lock_free() const noexcept;

#include <atomic>
#include <memory>

#include "../atomic_smart_ptr_test_types.h"
#include "test_macros.h"

template <class T>
void check() {
  using A = std::atomic<std::shared_ptr<T>>;

  static_assert(std::same_as<decltype(A::is_always_lock_free), const bool>);
  static_assert(!static_cast<bool>(A::is_always_lock_free));

  const A asp;
  std::same_as<bool> decltype(auto) lf = asp.is_lock_free();
  (void)lf;
  ASSERT_NOEXCEPT(A::is_always_lock_free);
  ASSERT_NOEXCEPT(asp.is_lock_free());
}

int main(int, char**) {
#define LIBCXX_ATOMIC_SP_RUN_IS_ALWAYS_LF(T) check<T>();
  LIBCXX_ATOMIC_SP_FOR_ALL_RUNTIME_TYPES(LIBCXX_ATOMIC_SP_RUN_IS_ALWAYS_LF)
#undef LIBCXX_ATOMIC_SP_RUN_IS_ALWAYS_LF
  return 0;
}
