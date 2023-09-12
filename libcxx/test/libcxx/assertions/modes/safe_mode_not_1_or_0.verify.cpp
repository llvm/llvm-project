//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test verifies that setting the safe mode to a value other than `0` or `1` triggers a compile-time error.

// Other hardening modes would additionally trigger the error that they are mutually exclusive.
// REQUIRES: libcpp-hardening-mode=unchecked
// Modules build produces a different error ("Could not build module 'std'").
// UNSUPPORTED: clang-modules-build
// ADDITIONAL_COMPILE_FLAGS: -Wno-macro-redefined -D_LIBCPP_ENABLE_SAFE_MODE=2

#include <cassert>

// expected-error@*:*  {{_LIBCPP_ENABLE_SAFE_MODE must be set to 0 or 1.}}
