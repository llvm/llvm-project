//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <__config>

// Ensure that defining _LIBCPP_ENABLE_CXX20_REMOVED_FEATURES yields a
// deprecation warning. We intend to issue a deprecation warning in LLVM 18
// and remove the macro entirely in LLVM 19. As such, this test will be quite
// short lived.

// UNSUPPORTED: clang-modules-build

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_CXX20_REMOVED_FEATURES

#include <version> // expected-warning@* 1+ {{macro '_LIBCPP_ENABLE_CXX20_REMOVED_FEATURES' has been marked as deprecated}}
