//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test verifies that setting the assertion semantic to a value that's not part of the predefined constants
// triggers a compile-time error.

// Modules build produces a different error ("Could not build module 'std'").
// UNSUPPORTED: clang-modules-build
// UNSUPPORTED: c++03, libcpp-has-no-experimental-hardening-observe-semantic
// REQUIRES: verify-support

// RUN: %{verify} -U_LIBCPP_ASSERTION_SEMANTIC -D_LIBCPP_ASSERTION_SEMANTIC=42
// `hardening-dependent` cannot be set as the semantic (it's only an indicator to use hardening-related logic to pick
// the final semantic).
// RUN: %{verify} -U_LIBCPP_ASSERTION_SEMANTIC -D_LIBCPP_ASSERTION_SEMANTIC=_LIBCPP_ASSERTION_SEMANTIC_HARDENING_DEPENDENT
// Make sure that common cases of misuse produce readable errors. We deliberately disallow setting the assertion
// semantic as if it were a boolean flag.
// RUN: %{verify} -U_LIBCPP_ASSERTION_SEMANTIC -D_LIBCPP_ASSERTION_SEMANTIC=0
// RUN: %{verify} -U_LIBCPP_ASSERTION_SEMANTIC -D_LIBCPP_ASSERTION_SEMANTIC=1
// RUN: %{verify} -U_LIBCPP_ASSERTION_SEMANTIC -D_LIBCPP_ASSERTION_SEMANTIC

#include <cassert>

// expected-error@*:* {{_LIBCPP_ASSERTION_SEMANTIC must be set to one of the following values: _LIBCPP_ASSERTION_SEMANTIC_IGNORE, _LIBCPP_ASSERTION_SEMANTIC_OBSERVE, _LIBCPP_ASSERTION_SEMANTIC_QUICK_ENFORCE, _LIBCPP_ASSERTION_SEMANTIC_ENFORCE}}
