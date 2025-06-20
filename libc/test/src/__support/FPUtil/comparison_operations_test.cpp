//===-- Unittests for Comparison Operations for FPBits class -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/ComparisonOperations.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/properties/types.h"
#include "src/__support/sign.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using LIBC_NAMESPACE::fputil::equals;
using LIBC_NAMESPACE::fputil::greater_than;
using LIBC_NAMESPACE::fputil::greater_than_or_equals;
using LIBC_NAMESPACE::fputil::less_than;
using LIBC_NAMESPACE::fputil::less_than_or_equals;
using LIBC_NAMESPACE::fputil::not_equals;

// FIXME: currently i have used NAN here
// need to find a better way to get a NAN floating point type
// - need to see if FPRep could be used?

#define TEST_EQUALS(Name, Type)                                                \
  TEST(LlvmLibcDyadic##Name##Test, Equals) {                                   \
    using Bits = LIBC_NAMESPACE::fputil::FPBits<Type>;                         \
    Type pos_zero = Bits::zero().get_val();                                    \
    Type neg_zero = -pos_zero;                                                 \
    Type pos_inf = Bits::inf().get_val();                                      \
    Type neg_inf = Bits::inf(Sign::NEG).get_val();                             \
    Type nan = NAN;                                                            \
    Type pos_normal = Type(3.14);                                              \
    Type neg_normal = Type(-2.71);                                             \
    Type pos_large = Type(1000000.0);                                          \
    Type neg_large = Type(-1000000.0);                                         \
                                                                               \
    EXPECT_TRUE(equals(pos_zero, pos_zero));                                   \
    EXPECT_TRUE(equals(neg_zero, neg_zero));                                   \
    EXPECT_TRUE(equals(pos_inf, pos_inf));                                     \
    EXPECT_TRUE(equals(neg_inf, neg_inf));                                     \
    EXPECT_TRUE(equals(pos_normal, pos_normal));                               \
    EXPECT_TRUE(equals(neg_normal, neg_normal));                               \
                                                                               \
    EXPECT_TRUE(equals(pos_zero, neg_zero));                                   \
    EXPECT_TRUE(equals(neg_zero, pos_zero));                                   \
                                                                               \
    EXPECT_FALSE(equals(pos_normal, neg_normal));                              \
    EXPECT_FALSE(equals(pos_normal, pos_large));                               \
    EXPECT_FALSE(equals(pos_inf, neg_inf));                                    \
    EXPECT_FALSE(equals(pos_inf, pos_normal));                                 \
    EXPECT_FALSE(equals(neg_inf, neg_normal));                                 \
    EXPECT_FALSE(equals(pos_large, neg_large));                                \
                                                                               \
    EXPECT_FALSE(equals(nan, nan));                                            \
    EXPECT_FALSE(equals(nan, pos_normal));                                     \
    EXPECT_FALSE(equals(nan, pos_zero));                                       \
    EXPECT_FALSE(equals(nan, pos_inf));                                        \
    EXPECT_FALSE(equals(pos_normal, nan));                                     \
  }

#define TEST_NOT_EQUALS(Name, Type)                                            \
  TEST(LlvmLibcDyadic##Name##Test, NotEquals) {                                \
    using Bits = LIBC_NAMESPACE::fputil::FPBits<Type>;                         \
    Type pos_zero = Bits::zero().get_val();                                    \
    Type neg_zero = Bits::zero(Sign::NEG).get_val();                           \
    Type pos_inf = Bits::inf().get_val();                                      \
    Type neg_inf = Bits::inf(Sign::NEG).get_val();                             \
    Type nan = NAN;                                                            \
    Type pos_normal = Type(3.14);                                              \
    Type neg_normal = Type(-2.71);                                             \
    Type pos_large = Type(1000000.0);                                          \
    Type neg_large = Type(-1000000.0);                                         \
                                                                               \
    EXPECT_FALSE(not_equals(pos_zero, pos_zero));                              \
    EXPECT_FALSE(not_equals(pos_zero, neg_zero));                              \
    EXPECT_FALSE(not_equals(pos_inf, pos_inf));                                \
    EXPECT_FALSE(not_equals(neg_inf, neg_inf));                                \
    EXPECT_FALSE(not_equals(pos_normal, pos_normal));                          \
                                                                               \
    EXPECT_TRUE(not_equals(pos_normal, neg_normal));                           \
    EXPECT_TRUE(not_equals(pos_inf, neg_inf));                                 \
    EXPECT_TRUE(not_equals(pos_normal, pos_zero));                             \
    EXPECT_TRUE(not_equals(pos_large, neg_large));                             \
    EXPECT_TRUE(not_equals(pos_inf, pos_normal));                              \
                                                                               \
    EXPECT_TRUE(not_equals(nan, nan));                                         \
    EXPECT_TRUE(not_equals(nan, pos_normal));                                  \
    EXPECT_TRUE(not_equals(nan, pos_zero));                                    \
    EXPECT_TRUE(not_equals(nan, pos_inf));                                     \
    EXPECT_TRUE(not_equals(pos_normal, nan));                                  \
  }

#define TEST_LESS_THAN(Name, Type)                                             \
  TEST(LlvmLibcDyadic##Name##Test, LessThan) {                                 \
    using Bits = LIBC_NAMESPACE::fputil::FPBits<Type>;                         \
    Type pos_zero = Bits::zero().get_val();                                    \
    Type neg_zero = -pos_zero;                                                 \
    Type pos_inf = Bits::inf().get_val();                                      \
    Type neg_inf = Bits::inf(Sign::NEG).get_val();                             \
    Type nan = NAN;                                                            \
    Type pos_small = Type(0.1);                                                \
    Type neg_small = Type(-0.1);                                               \
    Type pos_large = Type(1000000.0);                                          \
    Type neg_large = Type(-1000000.0);                                         \
                                                                               \
    EXPECT_TRUE(less_than(neg_small, pos_small));                              \
    EXPECT_TRUE(less_than(pos_small, pos_large));                              \
    EXPECT_TRUE(less_than(neg_large, neg_small));                              \
    EXPECT_FALSE(less_than(pos_large, pos_small));                             \
    EXPECT_FALSE(less_than(pos_small, neg_small));                             \
                                                                               \
    EXPECT_FALSE(less_than(pos_zero, neg_zero));                               \
    EXPECT_FALSE(less_than(neg_zero, pos_zero));                               \
    EXPECT_FALSE(less_than(pos_zero, pos_zero));                               \
                                                                               \
    EXPECT_TRUE(less_than(neg_small, pos_zero));                               \
    EXPECT_TRUE(less_than(neg_zero, pos_small));                               \
    EXPECT_FALSE(less_than(pos_small, pos_zero));                              \
                                                                               \
    EXPECT_TRUE(less_than(neg_inf, pos_inf));                                  \
    EXPECT_TRUE(less_than(neg_inf, neg_small));                                \
    EXPECT_TRUE(less_than(pos_small, pos_inf));                                \
    EXPECT_FALSE(less_than(pos_inf, pos_small));                               \
                                                                               \
    EXPECT_FALSE(less_than(pos_small, pos_small));                             \
    EXPECT_FALSE(less_than(neg_inf, neg_inf));                                 \
                                                                               \
    EXPECT_FALSE(less_than(nan, pos_small));                                   \
    EXPECT_FALSE(less_than(pos_small, nan));                                   \
    EXPECT_FALSE(less_than(nan, nan));                                         \
  }

#define TEST_GREATER_THAN(Name, Type)                                          \
  TEST(LlvmLibcDyadic##Name##Test, GreaterThan) {                              \
    using Bits = LIBC_NAMESPACE::fputil::FPBits<Type>;                         \
    Type pos_zero = Bits::zero().get_val();                                    \
    Type neg_zero = -pos_zero;                                                 \
    Type pos_inf = Bits::inf().get_val();                                      \
    Type neg_inf = Bits::inf(Sign::NEG).get_val();                             \
    Type nan = NAN;                                                            \
    Type pos_small = Type(0.1);                                                \
    Type neg_small = Type(-0.1);                                               \
    Type pos_large = Type(1000000.0);                                          \
    Type neg_large = Type(-1000000.0);                                         \
                                                                               \
    EXPECT_TRUE(greater_than(pos_small, neg_small));                           \
    EXPECT_TRUE(greater_than(pos_large, pos_small));                           \
    EXPECT_TRUE(greater_than(neg_small, neg_large));                           \
    EXPECT_FALSE(greater_than(pos_small, pos_large));                          \
    EXPECT_FALSE(greater_than(neg_small, pos_small));                          \
                                                                               \
    EXPECT_FALSE(greater_than(pos_zero, neg_zero));                            \
    EXPECT_FALSE(greater_than(neg_zero, pos_zero));                            \
                                                                               \
    EXPECT_TRUE(greater_than(pos_inf, neg_inf));                               \
    EXPECT_TRUE(greater_than(pos_inf, pos_small));                             \
    EXPECT_TRUE(greater_than(pos_small, neg_inf));                             \
    EXPECT_FALSE(greater_than(neg_inf, pos_inf));                              \
                                                                               \
    EXPECT_FALSE(greater_than(pos_small, pos_small));                          \
    EXPECT_FALSE(greater_than(pos_inf, pos_inf));                              \
                                                                               \
    EXPECT_FALSE(greater_than(nan, pos_small));                                \
    EXPECT_FALSE(greater_than(pos_small, nan));                                \
    EXPECT_FALSE(greater_than(nan, nan));                                      \
  }

#define TEST_LESS_THAN_OR_EQUALS(Name, Type)                                   \
  TEST(LlvmLibcDyadic##Name##Test, LessThanOrEquals) {                         \
    using Bits = LIBC_NAMESPACE::fputil::FPBits<Type>;                         \
    Type pos_zero = Bits::zero().get_val();                                    \
    Type neg_zero = -pos_zero;                                                 \
    Type pos_inf = Bits::inf().get_val();                                      \
    Type neg_inf = Bits::inf(Sign::NEG).get_val();                             \
    Type nan = NAN;                                                            \
    Type pos_small = Type(0.1);                                                \
    Type neg_small = Type(-0.1);                                               \
    Type pos_large = Type(1000000.0);                                          \
    Type neg_large = Type(-1000000.0);                                         \
                                                                               \
    EXPECT_TRUE(less_than_or_equals(neg_small, pos_small));                    \
    EXPECT_TRUE(less_than_or_equals(pos_small, pos_large));                    \
    EXPECT_TRUE(less_than_or_equals(neg_inf, pos_small));                      \
                                                                               \
    EXPECT_TRUE(less_than_or_equals(pos_small, pos_small));                    \
    EXPECT_TRUE(less_than_or_equals(pos_zero, neg_zero));                      \
    EXPECT_TRUE(less_than_or_equals(pos_inf, pos_inf));                        \
                                                                               \
    EXPECT_FALSE(less_than_or_equals(pos_small, neg_small));                   \
    EXPECT_FALSE(less_than_or_equals(pos_large, pos_small));                   \
    EXPECT_FALSE(less_than_or_equals(pos_inf, pos_small));                     \
                                                                               \
    EXPECT_TRUE(less_than_or_equals(neg_large, pos_small));                    \
    EXPECT_FALSE(less_than_or_equals(pos_large, neg_small));                   \
                                                                               \
    EXPECT_FALSE(less_than_or_equals(nan, pos_small));                         \
    EXPECT_FALSE(less_than_or_equals(pos_small, nan));                         \
    EXPECT_FALSE(less_than_or_equals(nan, nan));                               \
  }

#define TEST_GREATER_THAN_OR_EQUALS(Name, Type)                                \
  TEST(LlvmLibcDyadic##Name##Test, GreaterThanOrEquals) {                      \
    using Bits = LIBC_NAMESPACE::fputil::FPBits<Type>;                         \
    Type pos_zero = Bits::zero().get_val();                                    \
    Type neg_zero = -pos_zero;                                                 \
    Type pos_inf = Bits::inf().get_val();                                      \
    Type neg_inf = Bits::inf(Sign::NEG).get_val();                             \
    Type nan = NAN;                                                            \
    Type pos_small = Type(0.1);                                                \
    Type neg_small = Type(-0.1);                                               \
    Type pos_large = Type(1000000.0);                                          \
    Type neg_large = Type(-1000000.0);                                         \
                                                                               \
    EXPECT_TRUE(greater_than_or_equals(pos_small, neg_small));                 \
    EXPECT_TRUE(greater_than_or_equals(pos_large, pos_small));                 \
    EXPECT_TRUE(greater_than_or_equals(pos_inf, pos_small));                   \
                                                                               \
    EXPECT_TRUE(greater_than_or_equals(pos_small, pos_small));                 \
    EXPECT_TRUE(greater_than_or_equals(pos_zero, neg_zero));                   \
    EXPECT_TRUE(greater_than_or_equals(neg_inf, neg_inf));                     \
                                                                               \
    EXPECT_FALSE(greater_than_or_equals(neg_small, pos_small));                \
    EXPECT_FALSE(greater_than_or_equals(pos_small, pos_large));                \
    EXPECT_FALSE(greater_than_or_equals(neg_inf, pos_small));                  \
                                                                               \
    EXPECT_TRUE(greater_than_or_equals(pos_large, neg_small));                 \
    EXPECT_FALSE(greater_than_or_equals(neg_large, pos_small));                \
                                                                               \
    EXPECT_FALSE(greater_than_or_equals(nan, pos_small));                      \
    EXPECT_FALSE(greater_than_or_equals(pos_small, nan));                      \
    EXPECT_FALSE(greater_than_or_equals(nan, nan));                            \
  }

#define TEST_COMPARISON_OPS(Name, Type)                                        \
  TEST_EQUALS(Name, Type)                                                      \
  TEST_NOT_EQUALS(Name, Type)                                                  \
  TEST_LESS_THAN(Name, Type)                                                   \
  TEST_GREATER_THAN(Name, Type)                                                \
  TEST_LESS_THAN_OR_EQUALS(Name, Type)                                         \
  TEST_GREATER_THAN_OR_EQUALS(Name, Type)

TEST_COMPARISON_OPS(Float, float)
TEST_COMPARISON_OPS(Double, double)
// FIXME:  error: expected '(' for function-style cast or type construction
// TEST_COMPARISON_OPS(LongDouble, (long double))
#ifdef LIBC_TYPES_HAS_FLOAT16
TEST_COMPARISON_OPS(Float16, float16)
#endif // LIBC_TYPES_HAS_FLOAT16

// TODO: add other types if this is correct?
