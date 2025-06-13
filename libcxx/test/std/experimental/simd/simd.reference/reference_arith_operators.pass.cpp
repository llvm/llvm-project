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
// template<class U> reference+=(U&& x) && noexcept;
// template<class U> reference-=(U&& x) && noexcept;
// template<class U> reference*=(U&& x) && noexcept;
// template<class U> reference/=(U&& x) && noexcept;
// template<class U> reference%=(U&& x) && noexcept;

#include <experimental/simd>
#include <functional>

#include "../test_utils.h"

namespace ex = std::experimental::parallelism_v2;

struct PlusAssign {
  template <typename T, typename U>
  void operator()(T&& lhs, const U& rhs) const noexcept {
    std::forward<T>(lhs) += rhs;
  }
};

struct MinusAssign {
  template <typename T, typename U>
  void operator()(T&& lhs, const U& rhs) const noexcept {
    std::forward<T>(lhs) -= rhs;
  }
};

struct MultipliesAssign {
  template <typename T, typename U>
  void operator()(T&& lhs, const U& rhs) const noexcept {
    std::forward<T>(lhs) *= rhs;
  }
};

struct DividesAssign {
  template <typename T, typename U>
  void operator()(T&& lhs, const U& rhs) const noexcept {
    std::forward<T>(lhs) /= rhs;
  }
};

struct ModulusAssign {
  template <typename T, typename U>
  void operator()(T&& lhs, const U& rhs) const noexcept {
    std::forward<T>(lhs) %= rhs;
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

template <class T, std::size_t>
struct CheckReferenceArithOperators {
  template <class SimdAbi>
  void operator()() {
    types::for_each(simd_test_types(), SimdReferenceOperatorHelper<T, SimdAbi, std::plus<>, PlusAssign>());
    types::for_each(simd_test_types(), SimdReferenceOperatorHelper<T, SimdAbi, std::minus<>, MinusAssign>());
    types::for_each(simd_test_types(), SimdReferenceOperatorHelper<T, SimdAbi, std::multiplies<>, MultipliesAssign>());
    types::for_each(simd_test_types(), SimdReferenceOperatorHelper<T, SimdAbi, std::divides<>, DividesAssign>());
  }
};

template <class T, std::size_t>
struct CheckReferenceModOperators {
  template <class SimdAbi>
  void operator()() {
    types::for_each(
        simd_test_integer_types(), SimdReferenceOperatorHelper<T, SimdAbi, std::modulus<>, ModulusAssign>());
  }
};

int main(int, char**) {
  test_all_simd_abi<CheckReferenceArithOperators>();
  types::for_each(types::integer_types(), TestAllSimdAbiFunctor<CheckReferenceModOperators>());
  return 0;
}
