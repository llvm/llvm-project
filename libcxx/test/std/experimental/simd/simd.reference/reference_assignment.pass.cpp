//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// The machine emulated in tests does not have enough memory for code.
// UNSUPPORTED: LIBCXX-PICOLIBC-FIXME

// <experimental/simd>
//
// [simd.reference]
// template<class U> reference=(U&& x) && noexcept;
// XFAIL: target=powerpc{{.*}}le-unknown-linux-gnu

#include "../test_utils.h"
#include <experimental/simd>

namespace ex = std::experimental::parallelism_v2;

template <class T, class SimdAbi>
struct CheckSimdReferenceAssignmentHelper {
  template <class U>
  void operator()() const {
    if constexpr (std::is_assignable_v<T&, U&&>) {
      ex::simd<T, SimdAbi> origin_simd([](T i) { return i; });
      for (size_t i = 0; i < origin_simd.size(); ++i) {
        static_assert(noexcept(origin_simd[i] = static_cast<U>(i + 1)));
        origin_simd[i] = static_cast<U>(i + 1);
        assert(origin_simd[i] == static_cast<T>(std::forward<U>(i + 1)));
      }
    }
  }
};

template <class T, class SimdAbi>
struct CheckMaskReferenceAssignmentHelper {
  template <class U>
  void operator()() const {
    if constexpr (std::is_assignable_v<bool&, U&&>) {
      ex::simd_mask<T, SimdAbi> origin_mask(true);
      for (size_t i = 0; i < origin_mask.size(); ++i) {
        static_assert(noexcept(origin_mask[i] = static_cast<U>(i + 1)));
        origin_mask[i] = static_cast<U>(i % 2);
        assert(origin_mask[i] == static_cast<T>(std::forward<U>(i % 2)));
      }
    }
  }
};

template <class T, class SimdAbi>
struct CheckReferenceAssignmentTraitsHelper {
  template <class U>
  void operator()() const {
    if constexpr (std::is_assignable_v<T&, U&&>)
      static_assert(std::is_assignable_v<typename ex::simd<T, SimdAbi>::reference&&, U&&>);
    else
      static_assert(!std::is_assignable_v<typename ex::simd<T, SimdAbi>::reference&&, U&&>);

    if constexpr (std::is_assignable_v<bool&, U&&>)
      static_assert(std::is_assignable_v<typename ex::simd_mask<T, SimdAbi>::reference&&, U&&>);
    else
      static_assert(!std::is_assignable_v<typename ex::simd_mask<T, SimdAbi>::reference&&, U&&>);
  }
};

template <class T, std::size_t>
struct CheckReferenceAssignment {
  template <class SimdAbi>
  void operator()() {
    types::for_each(arithmetic_no_bool_types(), CheckSimdReferenceAssignmentHelper<T, SimdAbi>());
    types::for_each(arithmetic_no_bool_types(), CheckMaskReferenceAssignmentHelper<T, SimdAbi>());

    types::for_each(arithmetic_no_bool_types(), CheckReferenceAssignmentTraitsHelper<T, SimdAbi>());
  }
};

int main(int, char**) {
  test_all_simd_abi<CheckReferenceAssignment>();
  return 0;
}
