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
// template<class U> reference|=(U&& x) && noexcept;
// template<class U> reference&=(U&& x) && noexcept;
// template<class U> reference^=(U&& x) && noexcept;
// template<class U> reference<<=(U&& x) && noexcept;
// template<class U> reference>>=(U&& x) && noexcept;

#include "../test_utils.h"
#include <experimental/simd>
#include <iostream>

namespace ex = std::experimental::parallelism_v2;

#define LIBCXX_SIMD_REFERENCE_OP_(op, name)                                                                            \
  template <class T, class SimdAbi>                                                                                    \
  struct SimdReferenceOperatorHelper##name {                                                                           \
    template <class U>                                                                                                 \
    void operator()() const {                                                                                          \
      ex::simd<T, SimdAbi> origin_simd(2);                                                                             \
      for (size_t i = 0; i < origin_simd.size(); ++i) {                                                                \
        static_assert(noexcept(origin_simd[i] op## = static_cast<U>(i + 1)));                                          \
        origin_simd[i] op## = static_cast<U>(i + 1);                                                                   \
        assert((T)origin_simd[i] == (T)(static_cast<T>(2) op static_cast<T>(std::forward<U>(i + 1))));                 \
      }                                                                                                                \
    }                                                                                                                  \
  };
LIBCXX_SIMD_REFERENCE_OP_(+, Plus)
LIBCXX_SIMD_REFERENCE_OP_(-, Minus)
LIBCXX_SIMD_REFERENCE_OP_(*, Multiplies)
LIBCXX_SIMD_REFERENCE_OP_(/, Divides)
LIBCXX_SIMD_REFERENCE_OP_(%, Modulus)
LIBCXX_SIMD_REFERENCE_OP_(&, BitAnd)
LIBCXX_SIMD_REFERENCE_OP_(|, BitOr)
LIBCXX_SIMD_REFERENCE_OP_(^, BitXor)
LIBCXX_SIMD_REFERENCE_OP_(<<, ShiftLeft)
LIBCXX_SIMD_REFERENCE_OP_(>>, ShiftRight)
#undef LIBCXX_SIMD_REFERENCE_OP_

#define LIBCXX_SIMD_MASK_REFERENCE_OP_(op, name)                                                                       \
  template <class T, class SimdAbi>                                                                                    \
  struct MaskReferenceOperatorHelper##name {                                                                           \
    template <class U>                                                                                                 \
    void operator()() const {                                                                                          \
      ex::simd<T, SimdAbi> origin_simd_mask(true);                                                                     \
      for (size_t i = 0; i < origin_simd_mask.size(); ++i) {                                                           \
        static_assert(noexcept(origin_simd_mask[i] op## = static_cast<U>(i % 2)));                                     \
        origin_simd_mask[i] op## = static_cast<U>(i % 2);                                                              \
        assert((bool)origin_simd_mask[i] == (bool)(true op static_cast<bool>(std::forward<U>(i % 2))));                \
      }                                                                                                                \
    }                                                                                                                  \
  };
LIBCXX_SIMD_MASK_REFERENCE_OP_(&, BitAnd)
LIBCXX_SIMD_MASK_REFERENCE_OP_(|, BitOr)
LIBCXX_SIMD_MASK_REFERENCE_OP_(^, BitXor)
#undef LIBCXX_SIMD_MASK_REFERENCE_OP_

template <class T, std::size_t>
struct CheckReferenceArithOperators {
  template <class SimdAbi>
  void operator()() {
    types::for_each(simd_test_types(), SimdReferenceOperatorHelperPlus<T, SimdAbi>());
    types::for_each(simd_test_types(), SimdReferenceOperatorHelperMinus<T, SimdAbi>());
    types::for_each(simd_test_types(), SimdReferenceOperatorHelperMultiplies<T, SimdAbi>());
    types::for_each(simd_test_types(), SimdReferenceOperatorHelperDivides<T, SimdAbi>());
  }
};

template <class T, std::size_t>
struct CheckReferenceIntOperators {
  template <class SimdAbi>
  void operator()() {
    types::for_each(types::integer_types(), SimdReferenceOperatorHelperModulus<T, SimdAbi>());
    types::for_each(types::integer_types(), SimdReferenceOperatorHelperBitAnd<T, SimdAbi>());
    types::for_each(types::integer_types(), SimdReferenceOperatorHelperBitOr<T, SimdAbi>());
    types::for_each(types::integer_types(), SimdReferenceOperatorHelperBitXor<T, SimdAbi>());
    types::for_each(types::integer_types(), SimdReferenceOperatorHelperShiftLeft<T, SimdAbi>());
    types::for_each(types::integer_types(), SimdReferenceOperatorHelperShiftRight<T, SimdAbi>());

    types::for_each(types::integer_types(), MaskReferenceOperatorHelperBitAnd<T, SimdAbi>());
    types::for_each(types::integer_types(), MaskReferenceOperatorHelperBitOr<T, SimdAbi>());
    types::for_each(types::integer_types(), MaskReferenceOperatorHelperBitXor<T, SimdAbi>());
  }
};

int main(int, char**) {
  test_all_simd_abi<CheckReferenceArithOperators>();
  types::for_each(types::integer_types(), TestAllSimdAbiFunctor<CheckReferenceIntOperators>());
  return 0;
}
