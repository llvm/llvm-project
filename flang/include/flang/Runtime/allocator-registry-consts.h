//===-- include/flang/Runtime/allocator-registry-consts.h -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_ALLOCATOR_REGISTRY_CONSTS_H_
#define FORTRAN_RUNTIME_ALLOCATOR_REGISTRY_CONSTS_H_

RT_OFFLOAD_VAR_GROUP_BEGIN

static constexpr unsigned kDefaultAllocator = 0;

// Allocator used for CUF
static constexpr unsigned kPinnedAllocatorPos = 1;
static constexpr unsigned kDeviceAllocatorPos = 2;
static constexpr unsigned kManagedAllocatorPos = 3;
static constexpr unsigned kUnifiedAllocatorPos = 4;

RT_OFFLOAD_VAR_GROUP_END

#endif /* FORTRAN_RUNTIME_ALLOCATOR_REGISTRY_CONSTS_H_ */
