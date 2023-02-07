//===-- Convenient sanitizer macros -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_SANITIZER_H
#define LLVM_LIBC_SRC_SUPPORT_SANITIZER_H

#include "src/__support/macros/compiler_features.h"

// MemorySanitizer (MSan) is a detector of uninitialized reads. It consists of
// a compiler instrumentation module and a run-time library.
#ifdef LLVM_LIBC_HAVE_MEMORY_SANITIZER
#error "LLVM_LIBC_HAVE_MEMORY_SANITIZER cannot be directly set."
#elif defined(MEMORY_SANITIZER)
// The MEMORY_SANITIZER macro is deprecated but we will continue to honor it
// for now.
#define LLVM_LIBC_HAVE_MEMORY_SANITIZER 1
#elif defined(__SANITIZE_MEMORY__)
#define LLVM_LIBC_HAVE_MEMORY_SANITIZER 1
#elif !defined(__native_client__) && LLVM_LIBC_HAS_FEATURE(memory_sanitizer)
#define LLVM_LIBC_HAVE_MEMORY_SANITIZER 1
#endif

// AddressSanitizer (ASan) is a fast memory error detector.
#ifdef LLVM_LIBC_HAVE_ADDRESS_SANITIZER
#error "LLVM_LIBC_HAVE_ADDRESS_SANITIZER cannot be directly set."
#elif defined(ADDRESS_SANITIZER)
// The ADDRESS_SANITIZER macro is deprecated but we will continue to honor it
// for now.
#define LLVM_LIBC_HAVE_ADDRESS_SANITIZER 1
#elif defined(__SANITIZE_ADDRESS__)
#define LLVM_LIBC_HAVE_ADDRESS_SANITIZER 1
#elif LLVM_LIBC_HAS_FEATURE(address_sanitizer)
#define LLVM_LIBC_HAVE_ADDRESS_SANITIZER 1
#endif

// HWAddressSanitizer (HWASan) is a fast, low memory overhead error detector.
#ifdef LLVM_LIBC_HAVE_HWADDRESS_SANITIZER
#error "LLVM_LIBC_HAVE_HWADDRESS_SANITIZER cannot be directly set."
#elif LLVM_LIBC_HAS_FEATURE(hwaddress_sanitizer)
#define LLVM_LIBC_HAVE_HWADDRESS_SANITIZER 1
#endif

#if LLVM_LIBC_HAVE_MEMORY_SANITIZER
#include <sanitizer/msan_interface.h>
#define SANITIZER_MEMORY_INITIALIZED(addr, size) __msan_unpoison(addr, size)
#else
#define SANITIZER_MEMORY_INITIALIZED(ptr, size)
#endif

#endif // LLVM_LIBC_SRC_SUPPORT_SANITIZER_H
