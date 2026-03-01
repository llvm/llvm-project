//===-- lf_interface.h - LowFat Sanitizer Runtime Interface ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the LowFat Sanitizer runtime interface functions.
// The runtime library must define these functions so the instrumented program
// can call them.
//
//===----------------------------------------------------------------------===//
#ifndef LF_INTERFACE_H
#define LF_INTERFACE_H

#include "sanitizer_common/sanitizer_internal_defs.h"

using __sanitizer::uptr;

extern "C" {

// Initialize the LowFat Sanitizer runtime. Called early during program startup.
SANITIZER_INTERFACE_ATTRIBUTE void __lf_init();

// Report an out-of-bounds error.
// ptr: The pointer that caused the violation
// base: The base address of the allocation
// bound: The size of the allocation
SANITIZER_INTERFACE_ATTRIBUTE void __lf_report_oob(uptr ptr, uptr base,
                                                    uptr bound);

// Warn about an out-of-bounds error without terminating.
// ptr: The pointer that caused the violation
// base: The base address of the allocation
// bound: The size of the allocation
SANITIZER_INTERFACE_ATTRIBUTE void __lf_warn_oob(uptr ptr, uptr base,
                                                  uptr bound);

// Get the base address of an allocation from a pointer.
// Returns the base address, or 0 if the pointer is not within a LowFat region.
SANITIZER_INTERFACE_ATTRIBUTE uptr __lf_get_base(uptr ptr);

// Get the size (bound) of an allocation from a pointer.
// Returns the allocation size, or 0 if the pointer is not within a LowFat region.
SANITIZER_INTERFACE_ATTRIBUTE uptr __lf_get_size(uptr ptr);

}  // extern "C"

#endif  // LF_INTERFACE_H
