//===-- lf_allocator.h - LowFat Allocator Internal Interface ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Internal allocator interface shared between lf_rtl.cpp and
// lf_interceptors.cpp.
//
//===----------------------------------------------------------------------===//
#ifndef LF_ALLOCATOR_H
#define LF_ALLOCATOR_H

#include "sanitizer_common/sanitizer_internal_defs.h"

namespace __lowfat {

using __sanitizer::uptr;

// Allocate from a LowFat region. Returns nullptr if size exceeds max.
void *Allocate(uptr size);

// Free a LowFat allocation.
void Deallocate(void *ptr);

// Initialize interceptors (called from __lf_init).
void InitializeInterceptors();

} // namespace __lowfat

#endif // LF_ALLOCATOR_H
