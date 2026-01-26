//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test requires noexcept
// UNSUPPORTED: c++03

// GCC doesn't support -Wfunction-effect-analysis and the associated attributes
// UNSUPPORTED: gcc

// The observe semantic requires experimental support for now.
// XFAIL: libcpp-has-no-experimental-hardening-observe-semantic

// Make sure that libc++ assertions are compatible with the function effect analysis
// extension in Clang. The function effect analysis extension allows marking functions
// with attributes like [[nonblocking]], and Clang will issue a diagnostic if such a
// function calls another function that doesn't satisfy the requirements of the attribute.
//
// However, libc++'s assertion functions are called in a context where a precondition
// has been violated. Hence, it is acceptable not to satisfy these function effect
// attributes once the assertion is known to fail.
//
// This test ensures that we properly disable function effect analysis diagnostics
// in libc++'s assertion macros, otherwise it becomes impossible to call a function
// with hardened preconditions from e.g. a [[nonblocking]] function.

// ADDITIONAL_COMPILE_FLAGS: -Werror=function-effects

// RUN: %{verify}
// RUN: %{verify} -D_LIBCPP_ASSERTION_SEMANTIC=_LIBCPP_ASSERTION_SEMANTIC_IGNORE
// RUN: %{verify} -D_LIBCPP_ASSERTION_SEMANTIC=_LIBCPP_ASSERTION_SEMANTIC_OBSERVE
// RUN: %{verify} -D_LIBCPP_ASSERTION_SEMANTIC=_LIBCPP_ASSERTION_SEMANTIC_QUICK_ENFORCE
// RUN: %{verify} -D_LIBCPP_ASSERTION_SEMANTIC=_LIBCPP_ASSERTION_SEMANTIC_ENFORCE

#include <cstddef>
#include <span>

// Using _LIBCPP_ASSERT directly
void f(bool condition) noexcept [[clang::nonblocking]] {
  _LIBCPP_ASSERT(condition, "message"); // nothing
}
void g(bool condition) noexcept [[clang::nonallocating]] {
  _LIBCPP_ASSERT(condition, "message"); // nothing
}

// Sanity check with an actual std::span
void f(std::span<int> span, std::size_t index) noexcept [[clang::nonblocking]] {
  (void)span[index]; // nothing
}
void g(std::span<int> span, std::size_t index) noexcept [[clang::nonallocating]] {
  (void)span[index]; // nothing
}

// Test the test: ensure that a diagnostic would be emitted normally
void __potentially_blocking();
void f() noexcept [[clang::nonblocking]] {
  __potentially_blocking(); // expected-error {{function with 'nonblocking' attribute must not call non-'nonblocking' function}}
}
void g() noexcept [[clang::nonallocating]] {
  __potentially_blocking(); // expected-error {{function with 'nonallocating' attribute must not call non-'nonallocating' function}}
}
