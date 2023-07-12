//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test verifies that setting the hardened mode to a value other than `0` or `1` triggers a compile-time error.

// Debug mode would additionally trigger the error that hardened and debug modes are mutually exclusive.
// UNSUPPORTED: libcpp-has-debug-mode
// Modules build produces a different error ("Could not build module 'std'").
// UNSUPPORTED: modules-build
// ADDITIONAL_COMPILE_FLAGS: -Wno-macro-redefined -D_LIBCPP_ENABLE_HARDENED_MODE=2

#include <cassert>

// expected-error@*:*  {{_LIBCPP_ENABLE_HARDENED_MODE must be set to 0 or 1.}}
