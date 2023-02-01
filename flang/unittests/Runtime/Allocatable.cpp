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
  RTNAME(MoveAlloc)(*b, *a, false, nullptr, __FILE__, __LINE__);
  EXPECT_FALSE(a->IsAllocated());
  EXPECT_TRUE(b->IsAllocated());

  // move_alloc with stat
  std::int32_t stat{
      RTNAME(MoveAlloc)(*a, *b, true, nullptr, __FILE__, __LINE__)};
  EXPECT_TRUE(a->IsAllocated());
  EXPECT_FALSE(b->IsAllocated());
  EXPECT_EQ(stat, 0);

  // move_alloc with errMsg
  auto errMsg{Descriptor::Create(
      sizeof(char), 64, nullptr, 0, nullptr, CFI_attribute_allocatable)};
  errMsg->Allocate();
  RTNAME(MoveAlloc)(*b, *a, false, errMsg.get(), __FILE__, __LINE__);
  EXPECT_FALSE(a->IsAllocated());
  EXPECT_TRUE(b->IsAllocated());

  // move_alloc with stat and errMsg
  stat = RTNAME(MoveAlloc)(*a, *b, true, errMsg.get(), __FILE__, __LINE__);
  EXPECT_TRUE(a->IsAllocated());
  EXPECT_FALSE(b->IsAllocated());
  EXPECT_EQ(stat, 0);

  // move_alloc with the same deallocated array
  stat = RTNAME(MoveAlloc)(*b, *b, true, errMsg.get(), __FILE__, __LINE__);
  EXPECT_FALSE(b->IsAllocated());
  EXPECT_EQ(stat, 0);

  // move_alloc with the same allocated array should fail
  stat = RTNAME(MoveAlloc)(*a, *a, true, errMsg.get(), __FILE__, __LINE__);
  EXPECT_EQ(stat, 109);
  std::string_view errStr{errMsg->OffsetElement(), errMsg->ElementBytes()};
  auto trim_pos = errStr.find_last_not_of(' ');
  if (trim_pos != errStr.npos)
    errStr.remove_suffix(errStr.size() - trim_pos - 1);
  EXPECT_EQ(errStr, "MOVE_ALLOC passed the same address as to and from");
}
