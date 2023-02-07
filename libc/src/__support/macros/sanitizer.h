//===-- Convenient sanitizer macros -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_MACROS_SANITIZER_H
#define LLVM_LIBC_SRC_SUPPORT_MACROS_SANITIZER_H

#include "src/__support/macros/compiler_features.h"

//-----------------------------------------------------------------------------
// Properties to check the presence or absence or sanitizers
//-----------------------------------------------------------------------------

// MemorySanitizer (MSan) is a detector of uninitialized reads. It consists of
// a compiler instrumentation module and a run-time library. The
// MEMORY_SANITIZER macro is deprecated but we will continue to honor it for
// now.
#ifdef LIBC_HAVE_MEMORY_SANITIZER
#error "LIBC_HAVE_MEMORY_SANITIZER cannot be directly set."
#elif defined(MEMORY_SANITIZER) || defined(__SANITIZE_MEMORY__) ||             \
    (LLVM_LIBC_HAS_FEATURE(memory_sanitizer) && !defined(__native_client__))
#define LIBC_HAVE_MEMORY_SANITIZER
#endif

// AddressSanitizer (ASan) is a fast memory error detector. The
// ADDRESS_SANITIZER macro is deprecated but we will continue to honor it for
// now.
#ifdef LIBC_HAVE_ADDRESS_SANITIZER
#error "LIBC_HAVE_ADDRESS_SANITIZER cannot be directly set."
#elif defined(ADDRESS_SANITIZER) || defined(__SANITIZE_ADDRESS__) ||           \
    LLVM_LIBC_HAS_FEATURE(address_sanitizer)
#define LIBC_HAVE_ADDRESS_SANITIZER
#endif

// HWAddressSanitizer (HWASan) is a fast, low memory overhead error detector.
#ifdef LIBC_HAVE_HWADDRESS_SANITIZER
#error "LIBC_HAVE_HWADDRESS_SANITIZER cannot be directly set."
#elif LLVM_LIBC_HAS_FEATURE(hwaddress_sanitizer)
#define LIBC_HAVE_HWADDRESS_SANITIZER
#endif

//-----------------------------------------------------------------------------
// Functions to unpoison memory
//-----------------------------------------------------------------------------

#if LIBC_HAVE_MEMORY_SANITIZER
#include <sanitizer/msan_interface.h>
#define SANITIZER_MEMORY_INITIALIZED(addr, size) __msan_unpoison(addr, size)
#else
#define SANITIZER_MEMORY_INITIALIZED(ptr, size)
#endif
#endif // LLVM_LIBC_SRC_SUPPORT_MACROS_SANITIZER_H
