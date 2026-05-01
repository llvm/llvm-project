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
  using A = std::atomic<std::weak_ptr<T>>;

  static_assert(std::same_as<decltype(A::is_always_lock_free), const bool>);
  static_assert(A::is_always_lock_free == false);

  const A awp;
  std::same_as<bool> decltype(auto) lf = awp.is_lock_free();
  (void)lf;
  ASSERT_NOEXCEPT(A::is_always_lock_free);
  ASSERT_NOEXCEPT(awp.is_lock_free());
}

int main(int, char**) {
#define LIBCXX_ATOMIC_SP_RUN_W_IALF(T) check<T>();
  LIBCXX_ATOMIC_SP_FOR_ALL_RUNTIME_TYPES(LIBCXX_ATOMIC_SP_RUN_W_IALF)
#undef LIBCXX_ATOMIC_SP_RUN_W_IALF
  return 0;
}
