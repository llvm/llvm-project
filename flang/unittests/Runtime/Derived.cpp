//===-- flang/unittests/Runtime/Pointer.cpp--------- -------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "tools.h"
#include "flang/Runtime/derived-api.h"
#include "flang/Runtime/descriptor.h"

using namespace Fortran::runtime;

TEST(Derived, SameTypeAs) {
  // INTEGER, POINTER :: i1
  auto i1{
      Descriptor::Create(TypeCode{Fortran::common::TypeCategory::Integer, 4}, 4,
          nullptr, 0, nullptr, CFI_attribute_pointer)};
  EXPECT_TRUE(RTNAME(SameTypeAs)(*i1, *i1));

  auto r1{Descriptor::Create(TypeCode{Fortran::common::TypeCategory::Real, 4},
      4, nullptr, 0, nullptr, CFI_attribute_pointer)};
  EXPECT_FALSE(RTNAME(SameTypeAs)(*i1, *r1));

  // CLASS(*), ALLOCATABLE :: a1
  auto a1{Descriptor::Create(TypeCode{Fortran::common::TypeCategory::Real, 4},
      4, nullptr, 0, nullptr, CFI_attribute_allocatable)};
  a1->raw().elem_len = 0;
  a1->raw().type = CFI_type_other;

  EXPECT_FALSE(RTNAME(SameTypeAs)(*i1, *a1));
  EXPECT_FALSE(RTNAME(SameTypeAs)(*a1, *i1));
  EXPECT_FALSE(RTNAME(SameTypeAs)(*r1, *a1));

  // CLASS(*), ALLOCATABLE :: a2
  auto a2{Descriptor::Create(TypeCode{Fortran::common::TypeCategory::Real, 4},
      4, nullptr, 0, nullptr, CFI_attribute_allocatable)};
  a2->raw().elem_len = 0;
  a2->raw().type = CFI_type_other;

  EXPECT_FALSE(RTNAME(SameTypeAs)(*a1, *a2));

  // CLASS(*), POINTER :: p1
  auto p1{Descriptor::Create(TypeCode{Fortran::common::TypeCategory::Real, 4},
      4, nullptr, 0, nullptr, CFI_attribute_pointer)};
  p1->raw().elem_len = 0;
  p1->raw().type = CFI_type_other;

  EXPECT_FALSE(RTNAME(SameTypeAs)(*i1, *p1));
  EXPECT_FALSE(RTNAME(SameTypeAs)(*p1, *i1));
}
