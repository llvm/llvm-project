//===-- flang/unittests/Runtime/Allocatable.cpp--------- ---------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/allocatable.h"
#include "gtest/gtest.h"
#include "tools.h"

using namespace Fortran::runtime;

static OwningPtr<Descriptor> createAllocatable(
    Fortran::common::TypeCategory tc, int kind, int rank = 1) {
  return Descriptor::Create(TypeCode{tc, kind}, kind, nullptr, rank, nullptr,
      CFI_attribute_allocatable);
}

TEST(AllocatableTest, MoveAlloc) {
  using Fortran::common::TypeCategory;
  // INTEGER(4), ALLOCATABLE :: a(:)
  auto a{createAllocatable(TypeCategory::Integer, 4)};
  // INTEGER(4), ALLOCATABLE :: b(:)
  auto b{createAllocatable(TypeCategory::Integer, 4)};
  // ALLOCATE(a(20))
  a->GetDimension(0).SetBounds(1, 20);
  a->Allocate();

  EXPECT_TRUE(a->IsAllocated());
  EXPECT_FALSE(b->IsAllocated());

  // Simple move_alloc
  RTNAME(MoveAlloc)(*b, *a, nullptr, false, nullptr, __FILE__, __LINE__);
  EXPECT_FALSE(a->IsAllocated());
  EXPECT_TRUE(b->IsAllocated());

  // move_alloc with stat
  std::int32_t stat{
      RTNAME(MoveAlloc)(*a, *b, nullptr, true, nullptr, __FILE__, __LINE__)};
  EXPECT_TRUE(a->IsAllocated());
  EXPECT_FALSE(b->IsAllocated());
  EXPECT_EQ(stat, 0);

  // move_alloc with errMsg
  auto errMsg{Descriptor::Create(
      sizeof(char), 64, nullptr, 0, nullptr, CFI_attribute_allocatable)};
  errMsg->Allocate();
  RTNAME(MoveAlloc)(*b, *a, nullptr, false, errMsg.get(), __FILE__, __LINE__);
  EXPECT_FALSE(a->IsAllocated());
  EXPECT_TRUE(b->IsAllocated());

  // move_alloc with stat and errMsg
  stat = RTNAME(MoveAlloc)(
      *a, *b, nullptr, true, errMsg.get(), __FILE__, __LINE__);
  EXPECT_TRUE(a->IsAllocated());
  EXPECT_FALSE(b->IsAllocated());
  EXPECT_EQ(stat, 0);

  // move_alloc with the same deallocated array
  stat = RTNAME(MoveAlloc)(
      *b, *b, nullptr, true, errMsg.get(), __FILE__, __LINE__);
  EXPECT_FALSE(b->IsAllocated());
  EXPECT_EQ(stat, 0);

  // move_alloc with the same allocated array should fail
  stat = RTNAME(MoveAlloc)(
      *a, *a, nullptr, true, errMsg.get(), __FILE__, __LINE__);
  EXPECT_EQ(stat, 109);
  std::string_view errStr{errMsg->OffsetElement(), errMsg->ElementBytes()};
  auto trim_pos = errStr.find_last_not_of(' ');
  if (trim_pos != errStr.npos)
    errStr.remove_suffix(errStr.size() - trim_pos - 1);
  EXPECT_EQ(errStr, "MOVE_ALLOC passed the same address as to and from");
}

TEST(AllocatableTest, AllocateFromScalarSource) {
  using Fortran::common::TypeCategory;
  // REAL(4), ALLOCATABLE :: a(:)
  auto a{createAllocatable(TypeCategory::Real, 4)};
  // ALLOCATE(a(2:11), SOURCE=3.4)
  float sourecStorage{3.4F};
  auto s{Descriptor::Create(TypeCategory::Real, 4,
      reinterpret_cast<void *>(&sourecStorage), 0, nullptr,
      CFI_attribute_pointer)};
  RTNAME(AllocatableSetBounds)(*a, 0, 2, 11);
  RTNAME(AllocatableAllocateSource)
  (*a, *s, /*hasStat=*/false, /*errMsg=*/nullptr, __FILE__, __LINE__);
  EXPECT_TRUE(a->IsAllocated());
  EXPECT_EQ(a->Elements(), 10u);
  EXPECT_EQ(a->GetDimension(0).LowerBound(), 2);
  EXPECT_EQ(a->GetDimension(0).UpperBound(), 11);
  EXPECT_EQ(*a->OffsetElement<float>(), 3.4F);
  a->Destroy();
}

TEST(AllocatableTest, AllocateSourceZeroSize) {
  using Fortran::common::TypeCategory;
  // REAL(4), ALLOCATABLE :: a(:)
  auto a{createAllocatable(TypeCategory::Real, 4)};
  // REAL(4) :: s(-1:-2) = 0.
  float sourecStorage{0.F};
  const SubscriptValue extents[1]{0};
  auto s{Descriptor::Create(TypeCategory::Real, 4,
      reinterpret_cast<void *>(&sourecStorage), 1, extents,
      CFI_attribute_other)};
  // ALLOCATE(a, SOURCE=s)
  RTNAME(AllocatableSetBounds)(*a, 0, -1, -2);
  RTNAME(AllocatableAllocateSource)
  (*a, *s, /*hasStat=*/false, /*errMsg=*/nullptr, __FILE__, __LINE__);
  EXPECT_TRUE(a->IsAllocated());
  EXPECT_EQ(a->Elements(), 0u);
  EXPECT_EQ(a->GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(a->GetDimension(0).UpperBound(), 0);
  a->Destroy();
}

TEST(AllocatableTest, DoubleAllocation) {
  // CLASS(*), ALLOCATABLE :: r
  // ALLOCATE(REAL::r)
  auto r{createAllocatable(TypeCategory::Real, 4, 0)};
  EXPECT_FALSE(r->IsAllocated());
  EXPECT_TRUE(r->IsAllocatable());
  RTNAME(AllocatableAllocate)(*r);
  EXPECT_TRUE(r->IsAllocated());

  // Make sure AllocatableInitIntrinsicForAllocate doesn't reset the decsriptor
  // if it is allocated.
  // ALLOCATE(INTEGER::r)
  RTNAME(AllocatableInitIntrinsicForAllocate)
  (*r, Fortran::common::TypeCategory::Integer, 4);
  EXPECT_TRUE(r->IsAllocated());
}
