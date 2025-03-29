//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <experimental/simd>
//
// [simd.reference]
// template<class U> reference|=(U&& x) && noexcept;
// template<class U> reference&=(U&& x) && noexcept;
// template<class U> reference^=(U&& x) && noexcept;
// template<class U> reference<<=(U&& x) && noexcept;
// template<class U> reference>>=(U&& x) && noexcept;

#include <experimental/simd>
#include <functional>

#include "../test_utils.h"

namespace ex = std::experimental::parallelism_v2;

struct AndAssign {
  template <typename T, typename U>
  void operator()(T&& lhs, const U& rhs) const noexcept {
    std::forward<T>(lhs) &= rhs;
  }
};

struct OrAssign {
  template <typename T, typename U>
  void operator()(T&& lhs, const U& rhs) const noexcept {
    std::forward<T>(lhs) |= rhs;
  }
};

struct XorAssign {
  template <typename T, typename U>
  void operator()(T&& lhs, const U& rhs) const noexcept {
    std::forward<T>(lhs) ^= rhs;
  }
};

struct LeftShiftAssign {
  template <typename T, typename U>
  void operator()(T&& lhs, const U& rhs) const noexcept {
    std::forward<T>(lhs) <<= rhs;
  }
};

struct RightShiftAssign {
  template <typename T, typename U>
  void operator()(T&& lhs, const U& rhs) const noexcept {
    std::forward<T>(lhs) >>= rhs;
  }
};

struct LeftShift {
  template <typename T, typename U>
  T operator()(const T& lhs, const U& rhs) const noexcept {
    return lhs << rhs;
  }
};

struct RightShift {
  template <typename T, typename U>
  T operator()(const T& lhs, const U& rhs) const noexcept {
    return lhs >> rhs;
  }
};

template <typename T, typename SimdAbi, typename Op, typename OpAssign>
struct SimdReferenceOperatorHelper {
  template <class U>
  void operator()() const {
    ex::simd<T, SimdAbi> origin_simd(static_cast<T>(3));
    static_assert(noexcept(OpAssign{}(origin_simd[0], static_cast<U>(2))));
    OpAssign{}(origin_simd[0], static_cast<U>(2));
    assert((T)origin_simd[0] == (T)Op{}(static_cast<T>(3), static_cast<T>(std::forward<U>(2))));
  }
};

template <typename T, typename SimdAbi, typename Op, typename OpAssign>
struct MaskReferenceOperatorHelper {
  template <class U>
  void operator()() const {
    ex::simd_mask<T, SimdAbi> origin_mask(true);
    static_assert(noexcept(OpAssign{}(origin_mask[0], static_cast<U>(false))));
    OpAssign{}(origin_mask[0], static_cast<U>(false));
    assert((bool)origin_mask[0] == (bool)Op{}(true, static_cast<bool>(std::forward<U>(false))));
  }
};

template <class T, std::size_t>
struct CheckReferenceBitwiseOperators {
  template <class SimdAbi>
  void operator()() {
    types::for_each(simd_test_integer_types(), SimdReferenceOperatorHelper<T, SimdAbi, std::bit_and<>, AndAssign>());
    types::for_each(simd_test_integer_types(), SimdReferenceOperatorHelper<T, SimdAbi, std::bit_or<>, OrAssign>());
    types::for_each(simd_test_integer_types(), SimdReferenceOperatorHelper<T, SimdAbi, std::bit_xor<>, XorAssign>());
    types::for_each(simd_test_integer_types(), SimdReferenceOperatorHelper<T, SimdAbi, LeftShift, LeftShiftAssign>());
    types::for_each(simd_test_integer_types(), SimdReferenceOperatorHelper<T, SimdAbi, RightShift, RightShiftAssign>());

    types::for_each(simd_test_integer_types(), MaskReferenceOperatorHelper<T, SimdAbi, std::bit_and<>, AndAssign>());
    types::for_each(simd_test_integer_types(), MaskReferenceOperatorHelper<T, SimdAbi, std::bit_or<>, OrAssign>());
    types::for_each(simd_test_integer_types(), MaskReferenceOperatorHelper<T, SimdAbi, std::bit_xor<>, XorAssign>());
  }
};

int main(int, char**) {
  types::for_each(types::integer_types(), TestAllSimdAbiFunctor<CheckReferenceBitwiseOperators>());
  return 0;
}
