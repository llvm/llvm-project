//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test verifies that `_LIBCPP_ENABLE_HARDENED_MODE` and `_LIBCPP_ENABLE_DEBUG_MODE` are mutually exclusive.

// Modules build produces a different error ("Could not build module 'std'").
// UNSUPPORTED: modules-build
// ADDITIONAL_COMPILE_FLAGS: -Wno-macro-redefined -D_LIBCPP_ENABLE_HARDENED_MODE=1 -D_LIBCPP_ENABLE_DEBUG_MODE=1

#include <cassert>

// expected-error@*:*  {{Only one of _LIBCPP_ENABLE_HARDENED_MODE and _LIBCPP_ENABLE_DEBUG_MODE can be enabled.}}
