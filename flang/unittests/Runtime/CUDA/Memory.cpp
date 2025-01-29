//===-- flang/unittests/Runtime/Memory.cpp -----------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/CUDA/memory.h"
#include "gtest/gtest.h"
#include "../../../runtime/terminator.h"
#include "../tools.h"
#include "flang/Common/Fortran.h"
#include "flang/Runtime/CUDA/allocator.h"
#include "flang/Runtime/CUDA/common.h"
#include "flang/Runtime/CUDA/descriptor.h"
#include "flang/Runtime/allocatable.h"
#include "flang/Runtime/allocator-registry.h"

#include "cuda_runtime.h"

using namespace Fortran::runtime;
using namespace Fortran::runtime::cuda;

TEST(MemoryCUFTest, SimpleAllocTramsferFree) {
  int *dev = (int *)RTNAME(CUFMemAlloc)(
      sizeof(int), kMemTypeDevice, __FILE__, __LINE__);
  EXPECT_TRUE(dev != 0);
  int host = 42;
  RTNAME(CUFDataTransferPtrPtr)
  ((void *)dev, (void *)&host, sizeof(int), kHostToDevice, __FILE__, __LINE__);
  host = 0;
  RTNAME(CUFDataTransferPtrPtr)
  ((void *)&host, (void *)dev, sizeof(int), kDeviceToHost, __FILE__, __LINE__);
  EXPECT_EQ(42, host);
  RTNAME(CUFMemFree)((void *)dev, kMemTypeDevice, __FILE__, __LINE__);
}

static OwningPtr<Descriptor> createAllocatable(
    Fortran::common::TypeCategory tc, int kind, int rank = 1) {
  return Descriptor::Create(TypeCode{tc, kind}, kind, nullptr, rank, nullptr,
      CFI_attribute_allocatable);
}

TEST(MemoryCUFTest, CUFDataTransferDescDesc) {
  using Fortran::common::TypeCategory;
  RTNAME(CUFRegisterAllocator)();
  // INTEGER(4), DEVICE, ALLOCATABLE :: a(:)
  auto dev{createAllocatable(TypeCategory::Integer, 4)};
  dev->SetAllocIdx(kDeviceAllocatorPos);
  EXPECT_EQ((int)kDeviceAllocatorPos, dev->GetAllocIdx());
  RTNAME(AllocatableSetBounds)(*dev, 0, 1, 10);
  RTNAME(AllocatableAllocate)
  (*dev, /*hasStat=*/false, /*errMsg=*/nullptr, __FILE__, __LINE__);
  EXPECT_TRUE(dev->IsAllocated());

  // Create temp array to transfer to device.
  auto x{MakeArray<TypeCategory::Integer, 4>(std::vector<int>{10},
      std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9})};
  RTNAME(CUFDataTransferDescDesc)
  (dev.get(), x.get(), kHostToDevice, __FILE__, __LINE__);

  // Retrieve data from device.
  auto host{MakeArray<TypeCategory::Integer, 4>(std::vector<int>{10},
      std::vector<int32_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0})};
  RTNAME(CUFDataTransferDescDesc)
  (host.get(), dev.get(), kDeviceToHost, __FILE__, __LINE__);

  for (unsigned i = 0; i < 10; ++i) {
    EXPECT_EQ(*host->ZeroBasedIndexedElement<std::int32_t>(i), (std::int32_t)i);
  }
}
