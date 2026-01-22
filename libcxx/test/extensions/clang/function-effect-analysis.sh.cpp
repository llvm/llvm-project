//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

// ADDITIONAL_COMPILE_FLAGS: -Wfunction-effects -Werror=function-effects

// RUN: %{build}
// RUN: %{build} -D_LIBCPP_ASSERTION_SEMANTIC=_LIBCPP_ASSERTION_SEMANTIC_IGNORE
// RUN: %{build} -D_LIBCPP_ASSERTION_SEMANTIC=_LIBCPP_ASSERTION_SEMANTIC_OBSERVE
// RUN: %{build} -D_LIBCPP_ASSERTION_SEMANTIC=_LIBCPP_ASSERTION_SEMANTIC_QUICK_ENFORCE
// RUN: %{build} -D_LIBCPP_ASSERTION_SEMANTIC=_LIBCPP_ASSERTION_SEMANTIC_ENFORCE

#include <__assert>

void f(bool condition) noexcept [[clang::nonblocking]] { _LIBCPP_ASSERT(condition, "message"); }
void g(bool condition) noexcept [[clang::nonallocating]] { _LIBCPP_ASSERT(condition, "message"); }

int main(int, char**) { return 0; }
