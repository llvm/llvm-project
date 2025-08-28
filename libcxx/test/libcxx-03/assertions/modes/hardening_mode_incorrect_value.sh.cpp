//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test verifies that setting the hardening mode to a value that's not part of the predefined constants triggers
// a compile-time error.

// Modules build produces a different error ("Could not build module 'std'").
// UNSUPPORTED: clang-modules-build
// REQUIRES: verify-support

// Note that GCC doesn't support `-Wno-macro-redefined`.
// RUN: %{verify} -U_LIBCPP_HARDENING_MODE -D_LIBCPP_HARDENING_MODE=42
// Make sure that common cases of misuse produce readable errors. We deliberately disallow setting the hardening mode as
// if it were a boolean flag.
// RUN: %{verify} -U_LIBCPP_HARDENING_MODE -D_LIBCPP_HARDENING_MODE=0
// RUN: %{verify} -U_LIBCPP_HARDENING_MODE -D_LIBCPP_HARDENING_MODE=1
// RUN: %{verify} -U_LIBCPP_HARDENING_MODE -D_LIBCPP_HARDENING_MODE

#include <cassert>

// expected-error@*:* {{_LIBCPP_HARDENING_MODE must be set to one of the following values: _LIBCPP_HARDENING_MODE_NONE, _LIBCPP_HARDENING_MODE_FAST, _LIBCPP_HARDENING_MODE_EXTENSIVE, _LIBCPP_HARDENING_MODE_DEBUG}}
