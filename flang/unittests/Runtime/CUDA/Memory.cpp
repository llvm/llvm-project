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
#include "flang/Common/Fortran.h"
#include "flang/Runtime/CUDA/common.h"

#include "cuda_runtime.h"

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
