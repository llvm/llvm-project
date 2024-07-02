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

namespace ex = std::experimental::parallelism_v2;

#define LIBCXX_SIMD_REFERENCE_OP_(op, name)                                                                            \
  template <class T, class SimdAbi>                                                                                    \
  struct SimdReferenceOperatorHelper##name {                                                                           \
    template <class U>                                                                                                 \
    void operator()() const {                                                                                          \
      ex::simd<T, SimdAbi> origin_simd(static_cast<T>(3));                                                             \
      static_assert(noexcept(origin_simd[0] op## = static_cast<U>(2)));                                                \
      origin_simd[0] op## = static_cast<U>(2);                                                                         \
      assert((T)origin_simd[0] == (T)(static_cast<T>(3) op static_cast<T>(std::forward<U>(2))));                       \
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
      static_assert(noexcept(origin_simd_mask[0] op## = static_cast<U>(false)));                                       \
      origin_simd_mask[0] op## = static_cast<U>(false);                                                                \
      assert((bool)origin_simd_mask[0] == (bool)(true op static_cast<bool>(std::forward<U>(false))));                  \
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
    types::for_each(simd_test_integer_types(), SimdReferenceOperatorHelperModulus<T, SimdAbi>());
    types::for_each(simd_test_integer_types(), SimdReferenceOperatorHelperBitAnd<T, SimdAbi>());
    types::for_each(simd_test_integer_types(), SimdReferenceOperatorHelperBitOr<T, SimdAbi>());
    types::for_each(simd_test_integer_types(), SimdReferenceOperatorHelperBitXor<T, SimdAbi>());
    types::for_each(simd_test_integer_types(), SimdReferenceOperatorHelperShiftLeft<T, SimdAbi>());
    types::for_each(simd_test_integer_types(), SimdReferenceOperatorHelperShiftRight<T, SimdAbi>());

    types::for_each(simd_test_integer_types(), MaskReferenceOperatorHelperBitAnd<T, SimdAbi>());
    types::for_each(simd_test_integer_types(), MaskReferenceOperatorHelperBitOr<T, SimdAbi>());
    types::for_each(simd_test_integer_types(), MaskReferenceOperatorHelperBitXor<T, SimdAbi>());
  }
};

int main(int, char**) {
  test_all_simd_abi<CheckReferenceArithOperators>();
  types::for_each(types::integer_types(), TestAllSimdAbiFunctor<CheckReferenceIntOperators>());
  return 0;
}
