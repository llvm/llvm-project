//===-- asan_hsa_linux.h ---------------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Linux host HSA API interception for AddressSanitizer (SANITIZER_AMDHSA).
//
//===----------------------------------------------------------------------===//

#ifndef ASAN_HSA_LINUX_H
#define ASAN_HSA_LINUX_H

#include "asan_hsa_allocator.h"

#if SANITIZER_AMDHSA

namespace __asan {

void InitializeAmdgpuInterceptors();

}  // namespace __asan

#endif  // SANITIZER_AMDHSA

#endif  // ASAN_HSA_LINUX_H
