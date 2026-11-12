//===-- unittests/Runtime/CUDA/DefaultStream.cpp ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "flang/Runtime/CUDA/allocator.h"
#include "flang/Runtime/CUDA/stream.h"

using namespace Fortran::runtime;
using namespace Fortran::runtime::cuda;

TEST(DefaultStreamTest, GetAndSetTest) {
  using Fortran::common::TypeCategory;
  cudaStream_t defaultStream = RTDECL(CUFGetDefaultStream)();
  EXPECT_EQ(defaultStream, nullptr);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  EXPECT_EQ(cudaSuccess, cudaGetLastError());
  RTDECL(CUFSetDefaultStream)(stream);
  cudaStream_t outStream = RTDECL(CUFGetDefaultStream)();
  EXPECT_EQ(outStream, stream);
}

TEST(DefaultStreamTest, GetAndSetArrayTest) {
  using Fortran::common::TypeCategory;
  cudaStream_t defaultStream = RTDECL(CUFGetDefaultStream)();
  EXPECT_EQ(defaultStream, nullptr);

  cudaStream_t outStream = RTDECL(CUFGetDefaultStream)();
  EXPECT_EQ(outStream, nullptr);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  EXPECT_EQ(cudaSuccess, cudaGetLastError());
  RTDECL(CUFSetDefaultStream)(stream);
  outStream = RTDECL(CUFGetDefaultStream)();
  EXPECT_EQ(outStream, stream);
}
