//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure that we diagnose any usage of the default verbose termination function
// on a platform that doesn't support it at compile-time.

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_ASSERTIONS=1

// REQUIRES: use_system_cxx_lib && target={{.+}}-apple-macosx{{10.9|10.10|10.11|10.12|10.13|10.14|10.15|11|12}}

#include <version> // any header would work

void f() {
  _LIBCPP_ASSERT(true, "message"); // expected-error {{'__libcpp_verbose_abort' is unavailable}}
}
