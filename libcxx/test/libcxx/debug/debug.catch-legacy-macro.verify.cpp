//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test ensures that we issue an error if we try to enable the debug mode with
// a library that was not built with support for the debug mode.

// REQUIRES: !libcpp-has-debug-mode
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1

// This test fails when modules are enabled because we fail to build module 'std' instead of
// issuing the preprocessor error.
// UNSUPPORTED: modules-build

#include <__debug>

// expected-error@*:* {{Enabling the debug mode now requires having configured the library with support for the debug mode}}
