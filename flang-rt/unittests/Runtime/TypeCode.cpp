//===-- unittests/Runtime/TypeCode.cpp --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "flang-rt/runtime/type-code.h"

using namespace Fortran::runtime;
using namespace Fortran::common;

TEST(TypeCode, ComplexTypes) {
  // Test all Complex type kinds to ensure they map correctly
  struct ComplexTypeMapping {
    int kind;
    Fortran::ISO::CFI_type_t expectedType;
  };

  ComplexTypeMapping mappings[] = {
      {2, CFI_type_half_float_Complex},
      {3, CFI_type_bfloat_Complex},
      {4, CFI_type_float_Complex},
      {8, CFI_type_double_Complex},
      {10, CFI_type_extended_double_Complex},
      {16, CFI_type_float128_Complex},
  };

  for (const auto &mapping : mappings) {
    TypeCode tc(TypeCategory::Complex, mapping.kind);
    EXPECT_EQ(tc.raw(), mapping.expectedType)
        << "Complex kind " << mapping.kind << " should map to CFI type "
        << mapping.expectedType;
    EXPECT_TRUE(tc.IsComplex());

    auto categoryAndKind = tc.GetCategoryAndKind();
    ASSERT_TRUE(categoryAndKind.has_value());
    EXPECT_EQ(categoryAndKind->first, TypeCategory::Complex);
    EXPECT_EQ(categoryAndKind->second, mapping.kind);
  }
}
