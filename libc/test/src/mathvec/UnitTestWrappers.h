//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unit test logic for SIMD math functions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATHVEC_UNITTESTWRAPPERS_H
#define LLVM_LIBC_TEST_SRC_MATHVEC_UNITTESTWRAPPERS_H

#include "hdr/stdint_proxy.h"
#include "src/__support/CPP/simd.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "test/UnitTest/SIMDMatcher.h"

#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {
namespace testing {
namespace mathvec {

template <typename T> using UnaryScalarFunc = T(T);
template <typename T> using UnaryVectorFunc = cpp::simd<T>(cpp::simd<T>);

template <typename T, UnaryScalarFunc<T> ScalarFunc,
          UnaryVectorFunc<T> VectorFunc>
struct UnaryOp {
  using ScalarType = T;
  using VectorType = cpp::simd<T>;

  LIBC_INLINE static VectorType eval_scalar(VectorType input) {
    VectorType output = input;
    constexpr size_t N = cpp::internal::native_vector_size<T>;
    for (size_t i = 0; i < N; ++i)
      output[i] = ScalarFunc(input[i]);
    return output;
  }

  LIBC_INLINE static VectorType eval_vector(VectorType input) {
    return VectorFunc(input);
  }
};

namespace internal {

template <typename Op>
LIBC_INLINE typename Op::VectorType
make_input(typename Op::ScalarType x, typename Op::ScalarType control) {
  typename Op::VectorType input(x);
  input[0] = control;
  return input;
}

} // namespace internal

template <typename Op>
LIBC_INLINE typename Op::VectorType wrap_ref(typename Op::ScalarType x,
                                             typename Op::ScalarType control) {
  return Op::eval_scalar(internal::make_input<Op>(x, control));
}

template <typename Op>
LIBC_INLINE typename Op::VectorType wrap_ref(typename Op::ScalarType x) {
  return wrap_ref<Op>(x, x);
}

template <typename Op>
LIBC_INLINE typename Op::VectorType
wrap_vector(typename Op::ScalarType x, typename Op::ScalarType control) {
  return Op::eval_vector(internal::make_input<Op>(x, control));
}

template <typename Op>
LIBC_INLINE typename Op::VectorType wrap_vector(typename Op::ScalarType x) {
  return wrap_vector<Op>(x, x);
}

// A helper macro to test a variety of cases for a given input and operation.
// Passes in various edge cases as the control input to ensure that vectorized
// implementations behaves correctly across different input types.
#define TEST_VARIED_CASES(x, Op)                                               \
  do {                                                                         \
    EXPECT_SIMD_EQ(wrap_ref<Op>((x), -(x)), wrap_vector<Op>((x), -(x)));       \
    EXPECT_SIMD_EQ(wrap_ref<Op>(-(x), (x)), wrap_vector<Op>(-(x), (x)));       \
    EXPECT_SIMD_EQ(wrap_ref<Op>((x), aNaN), wrap_vector<Op>((x), aNaN));       \
    EXPECT_SIMD_EQ(wrap_ref<Op>((x), inf), wrap_vector<Op>((x), inf));         \
    EXPECT_SIMD_EQ(wrap_ref<Op>((x), neg_inf), wrap_vector<Op>((x), neg_inf)); \
    EXPECT_SIMD_EQ(wrap_ref<Op>((x), 0.0), wrap_vector<Op>((x), 0.0));         \
    EXPECT_SIMD_EQ(wrap_ref<Op>((x), -0.0), wrap_vector<Op>((x), -0.0));       \
    EXPECT_SIMD_EQ(wrap_ref<Op>((x), 1.0), wrap_vector<Op>((x), 1.0));         \
  } while (0)

// A helper macro to test a full range of float values, from 0 to 0x7f800000.
// Negative values are tested via the TEST_VARIED_CASES macro.
// The number of values to test is controlled by the LIBC_TEST_FLOAT_RANGE_COUNT
// macro, which can be set to any positive integer value.
#define TEST_MATHVEC_FLOAT_RANGE(Op)                                           \
  do {                                                                         \
    constexpr uint32_t COUNT = LIBC_TEST_FLOAT_RANGE_COUNT;                    \
    constexpr uint32_t RANGE = 0x7f800000U;                                    \
    constexpr uint32_t STEP = (RANGE / COUNT) > 0 ? (RANGE / COUNT) : 1;       \
    for (uint32_t i = 0, v = 0; i <= COUNT; ++i, v += STEP) {                  \
      float x = FPBits(v).get_val();                                           \
      TEST_VARIED_CASES(x, Op);                                                \
    }                                                                          \
  } while (0)

} // namespace mathvec
} // namespace testing
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_TEST_SRC_MATHVEC_UNITTESTWRAPPERS_H
