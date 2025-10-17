//===-- Portable attributes -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This header file defines macros for declaring attributes for functions,
// types, and variables.
//
// These macros are used within llvm-libc and allow the compiler to optimize,
// where applicable, certain function calls.
//
// Most macros here are exposing GCC or Clang features, and are stubbed out for
// other compilers.

#ifndef LLVM_LIBC_SRC___SUPPORT_MACROS_ATTRIBUTES_H
#define LLVM_LIBC_SRC___SUPPORT_MACROS_ATTRIBUTES_H

#include "config.h"
#include "properties/architectures.h"

#ifndef __has_attribute
#define __has_attribute(x) 0
#endif

#define LIBC_INLINE inline
#define LIBC_INLINE_VAR inline
#define LIBC_INLINE_ASM __asm__ __volatile__
#define LIBC_UNUSED __attribute__((unused))

// Uses the platform specific specialization
#define LIBC_THREAD_MODE_PLATFORM 0

// Mutex guards nothing, used in single-threaded implementations
#define LIBC_THREAD_MODE_SINGLE 1

// Vendor provides implementation
#define LIBC_THREAD_MODE_EXTERNAL 2

// libcxx doesn't define LIBC_THREAD_MODE, unless that is passed in the command
// line in the CMake invocation. This defaults to the original implementation
// (before changes in https://github.com/llvm/llvm-project/pull/145358)
#ifndef LIBC_THREAD_MODE
#define LIBC_THREAD_MODE LIBC_THREAD_MODE_PLATFORM
#endif // LIBC_THREAD_MODE

#if LIBC_THREAD_MODE != LIBC_THREAD_MODE_PLATFORM &&                           \
    LIBC_THREAD_MODE != LIBC_THREAD_MODE_SINGLE &&                             \
    LIBC_THREAD_MODE != LIBC_THREAD_MODE_EXTERNAL
#error LIBC_THREAD_MODE must be one of the following values: \
LIBC_THREAD_MODE_PLATFORM, \
LIBC_THREAD_MODE_SINGLE, \
LIBC_THREAD_MODE_EXTERNAL.
#endif

#if LIBC_THREAD_MODE == LIBC_THREAD_MODE_SINGLE
#define LIBC_THREAD_LOCAL
#else
#define LIBC_THREAD_LOCAL thread_local
#endif

#if __cplusplus >= 202002L
#define LIBC_CONSTINIT constinit
#elif __has_attribute(__require_constant_initialization__)
#define LIBC_CONSTINIT __attribute__((__require_constant_initialization__))
#else
#define LIBC_CONSTINIT
#endif

#if defined(__clang__) && __has_attribute(preferred_type)
#define LIBC_PREFERED_TYPE(TYPE) [[clang::preferred_type(TYPE)]]
#else
#define LIBC_PREFERED_TYPE(TYPE)
#endif

#if __has_attribute(ext_vector_type) &&                                        \
    LIBC_HAS_FEATURE(ext_vector_type_boolean)
#define LIBC_HAS_VECTOR_TYPE 1
#else
#define LIBC_HAS_VECTOR_TYPE 0
#endif

#if __has_attribute(no_sanitize)
// Disable regular and hardware-supported ASan for functions that may
// intentionally make out-of-bounds access. Disable TSan as well, as it detects
// out-of-bounds accesses to heap memory.
#define LIBC_NO_SANITIZE_OOB_ACCESS                                            \
  __attribute__((no_sanitize("address", "hwaddress", "thread")))
#else
#define LIBC_NO_SANITIZE_OOB_ACCESS
#endif

// LIBC_LIFETIME_BOUND indicates that a function parameter's lifetime is tied
// to the return value. This helps compilers detect use-after-free bugs.
//
// Example usage:
//   const T &get_value(const Container &c LIBC_LIFETIME_BOUND,
///                      const T &default_val LIBC_LIFETIME_BOUND);
//   // Warns if temporary Container or default_val is bound to the result
//
// For member functions, apply after the function signature:
//   const char *data() const LIBC_LIFETIME_BOUND;
//   // The returned pointer should not outlive '*this'
#if __has_cpp_attribute(clang::lifetimebound)
#define LIBC_LIFETIME_BOUND [[clang::lifetimebound]]
#elif __has_cpp_attribute(msvc::lifetimebound)
#define LIBC_LIFETIME_BOUND [[msvc::lifetimebound]]
#elif __has_cpp_attribute(lifetimebound)
#define LIBC_LIFETIME_BOUND [[lifetimebound]]
#else
#define LIBC_LIFETIME_BOUND
#endif

// LIBC_LIFETIME_CAPTURE_BY(X) indicates that parameter X captures/stores a
// reference to the annotated parameter. Warns if temporaries are passed.
//
// Example usage:
//   void add_to_set(cpp::string_view a LIBC_LIFETIME_CAPTURE_BY(s),
//                   cpp::set<cpp::string_view>& s) {
//     s.insert(a); // 's' now holds a reference to 'a'
//   }
//   // Warns: add_to_set(cpp::string(), s); // temporary captured by 's'
//
// X can be: another parameter name, 'this', 'global', or 'unknown'
// Multiple capturing entities: LIBC_LIFETIME_CAPTURE_BY(s1, s2)
//
// For member functions capturing 'this', apply after function signature:
//   void capture_self(cpp::set<S*>& s) LIBC_LIFETIME_CAPTURE_BY(s);
#if __has_cpp_attribute(clang::lifetime_capture_by)
#define LIBC_LIFETIME_CAPTURE_BY(X) [[clang::lifetime_capture_by(X)]]
#else
#define LIBC_LIFETIME_CAPTURE_BY(X)
#endif

// LIBC_GSL_POINTER marks a class as a "view" type that points to data owned
// elsewhere. Lifetime analysis treats it as potentially dangling when the
// owner is destroyed. Use for types like string_view, span, or custom views.
//
// Example usage:
//   class LIBC_GSL_POINTER StringView {
//     const char *data_;
//   public:
//     StringView(const String& s); // Points into 's'
//   };
//   // Warns: StringView sv = String(); // sv points to destroyed temporary
//
// The attribute takes an optional type parameter (e.g., [[gsl::Pointer(int)]])
// but it's typically omitted in libc usage.
#if __has_cpp_attribute(gsl::Pointer)
#define LIBC_GSL_POINTER [[gsl::Pointer]]
#else
#define LIBC_GSL_POINTER
#endif

// LIBC_GSL_OWNER marks a class as owning the data it manages. When an Owner
// is destroyed, any Pointer constructed from it becomes dangling.
//
// Example usage:
//   class LIBC_GSL_OWNER String {
//     char *data_;
//   public:
//     ~String() { delete[] data_; }
//   };
//
// Relationship: LIBC_GSL_POINTER types "point into" LIBC_GSL_OWNER types.
// When the owner dies, pointers derived from it are considered dangling.
#if __has_cpp_attribute(gsl::Owner)
#define LIBC_GSL_OWNER [[gsl::Owner]]
#else
#define LIBC_GSL_OWNER
#endif

#endif // LLVM_LIBC_SRC___SUPPORT_MACROS_ATTRIBUTES_H
