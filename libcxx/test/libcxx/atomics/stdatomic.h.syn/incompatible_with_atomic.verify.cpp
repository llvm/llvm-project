//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-has-no-threads
// REQUIRES: c++03 || c++11 || c++14 || c++17 || c++20

// This test ensures that we issue a reasonable diagnostic when using <stdatomic.h> while <atomic>
// is in use too. Before C++23, this otherwise leads to obscure errors because <stdatomic.h> tries
// to redefine things defined by <atomic>.

// Ignore additional weird errors that happen when the two headers are mixed.
// ADDITIONAL_COMPILE_FLAGS: -Xclang -verify-ignore-unexpected=error -Xclang -verify-ignore-unexpected=warning

#include <atomic>
#include <stdatomic.h>

// expected-error@*:* {{<stdatomic.h> is incompatible with <atomic> before C++23}}
