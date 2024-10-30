//===-- flang/unittests/Runtime/AllocatableCUF.cpp ---------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "../../../runtime/terminator.h"
#include "flang/Common/Fortran.h"
#include "flang/Runtime/CUDA/allocator.h"
#include "flang/Runtime/CUDA/descriptor.h"
#include "flang/Runtime/allocatable.h"
#include "flang/Runtime/allocator-registry.h"

#include "cuda_runtime.h"

using namespace Fortran::runtime;
using namespace Fortran::runtime::cuda;

static OwningPtr<Descriptor> createAllocatable(
    Fortran::common::TypeCategory tc, int kind, int rank = 1) {
  return Descriptor::Create(TypeCode{tc, kind}, kind, nullptr, rank, nullptr,
      CFI_attribute_allocatable);
}

TEST(AllocatableCUFTest, SimpleDeviceAllocate) {
  using Fortran::common::TypeCategory;
  RTNAME(CUFRegisterAllocator)();
  // REAL(4), DEVICE, ALLOCATABLE :: a(:)
  auto a{createAllocatable(TypeCategory::Real, 4)};
  a->SetAllocIdx(kDeviceAllocatorPos);
  EXPECT_EQ((int)kDeviceAllocatorPos, a->GetAllocIdx());
  EXPECT_FALSE(a->HasAddendum());
  RTNAME(AllocatableSetBounds)(*a, 0, 1, 10);
  RTNAME(AllocatableAllocate)
  (*a, /*hasStat=*/false, /*errMsg=*/nullptr, __FILE__, __LINE__);
  EXPECT_TRUE(a->IsAllocated());
  RTNAME(AllocatableDeallocate)
  (*a, /*hasStat=*/false, /*errMsg=*/nullptr, __FILE__, __LINE__);
  EXPECT_FALSE(a->IsAllocated());
}

TEST(AllocatableCUFTest, SimplePinnedAllocate) {
  using Fortran::common::TypeCategory;
  RTNAME(CUFRegisterAllocator)();
  // INTEGER(4), PINNED, ALLOCATABLE :: a(:)
  auto a{createAllocatable(TypeCategory::Integer, 4)};
  EXPECT_FALSE(a->HasAddendum());
  a->SetAllocIdx(kPinnedAllocatorPos);
  EXPECT_EQ((int)kPinnedAllocatorPos, a->GetAllocIdx());
  EXPECT_FALSE(a->HasAddendum());
  RTNAME(AllocatableSetBounds)(*a, 0, 1, 10);
  RTNAME(AllocatableAllocate)
  (*a, /*hasStat=*/false, /*errMsg=*/nullptr, __FILE__, __LINE__);
  EXPECT_TRUE(a->IsAllocated());
  RTNAME(AllocatableDeallocate)
  (*a, /*hasStat=*/false, /*errMsg=*/nullptr, __FILE__, __LINE__);
  EXPECT_FALSE(a->IsAllocated());
}

TEST(AllocatableCUFTest, DescriptorAllocationTest) {
  using Fortran::common::TypeCategory;
  RTNAME(CUFRegisterAllocator)();
  // REAL(4), DEVICE, ALLOCATABLE :: a(:)
  auto a{createAllocatable(TypeCategory::Real, 4)};
  Descriptor *desc = nullptr;
  desc = RTNAME(CUFAllocDesciptor)(a->SizeInBytes());
  EXPECT_TRUE(desc != nullptr);
  RTNAME(CUFFreeDesciptor)(desc);
}
