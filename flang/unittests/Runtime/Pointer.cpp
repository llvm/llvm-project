//===-- flang/unittests/Runtime/Pointer.cpp--------- -------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/pointer.h"
#include "gtest/gtest.h"
#include "tools.h"
#include "flang/Runtime/descriptor.h"

using namespace Fortran::runtime;

TEST(Pointer, BasicAllocateDeallocate) {
  // REAL(4), POINTER :: p(:)
  auto p{Descriptor::Create(TypeCode{Fortran::common::TypeCategory::Real, 4}, 4,
      nullptr, 1, nullptr, CFI_attribute_pointer)};
  // ALLOCATE(p(2:11))
  RTNAME(PointerSetBounds)(*p, 0, 2, 11);
  RTNAME(PointerAllocate)
  (*p, /*hasStat=*/false, /*errMsg=*/nullptr, __FILE__, __LINE__);
  EXPECT_TRUE(RTNAME(PointerIsAssociated)(*p));
  EXPECT_EQ(p->Elements(), 10u);
  EXPECT_EQ(p->GetDimension(0).LowerBound(), 2);
  EXPECT_EQ(p->GetDimension(0).UpperBound(), 11);
  // DEALLOCATE(p)
  RTNAME(PointerDeallocate)
  (*p, /*hasStat=*/false, /*errMsg=*/nullptr, __FILE__, __LINE__);
  EXPECT_FALSE(RTNAME(PointerIsAssociated)(*p));
}
