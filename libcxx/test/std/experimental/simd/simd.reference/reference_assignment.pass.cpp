//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// FIXME: Timeouts.
// UNSUPPORTED: sanitizer-new-delete

// <experimental/simd>
//
// [simd.reference]
// template<class U> reference=(U&& x) && noexcept;
//
// XFAIL: LIBCXX-AIX-FIXME

#include "../test_utils.h"
#include <experimental/simd>

namespace ex = std::experimental::parallelism_v2;

template <class T, class SimdAbi>
struct CheckSimdReferenceAssignmentHelper {
  template <class U>
  void operator()() const {
    if constexpr (std::is_assignable_v<T&, U&&>) {
      ex::simd<T, SimdAbi> origin_simd([](T i) { return i; });
      static_assert(noexcept(origin_simd[0] = static_cast<U>(5)));
      origin_simd[0] = static_cast<U>(5);
      assert(origin_simd[0] == static_cast<T>(std::forward<U>(5)));
    }
  }
};

template <class T, class SimdAbi>
struct CheckMaskReferenceAssignmentHelper {
  template <class U>
  void operator()() const {
    if constexpr (std::is_assignable_v<bool&, U&&>) {
      ex::simd_mask<T, SimdAbi> origin_mask(true);
      static_assert(noexcept(origin_mask[0] = false));
      origin_mask[0] = false;
      assert(origin_mask[0] == false);
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
