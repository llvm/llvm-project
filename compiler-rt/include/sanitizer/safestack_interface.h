//===-- sanitizer/safestack_interface.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of SafeStack.
//
// Public interface header.
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_SAFESTACK_INTERFACE_H
#define SANITIZER_SAFESTACK_INTERFACE_H

#include <sanitizer/common_interface_defs.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Returns the current unsafe stack pointer of the current thread.
const void * SANITIZER_CDECL __safestack_get_unsafe_stack_ptr(void);

/// Returns a pointer to the bottom of the unsafe stack of the current thread.
const void * SANITIZER_CDECL __safestack_get_unsafe_stack_bottom(void);

/// Returns a pointer to the top of the unsafe stack of the current thread.
const void * SANITIZER_CDECL __safestack_get_unsafe_stack_top(void);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // SANITIZER_SAFESTACK_INTERFACE_H
