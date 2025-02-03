//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads
// REQUIRES: c++03 || c++11 || c++14 || c++17 || c++20

// This test ensures that we issue a reasonable diagnostic when including <atomic> after
// <stdatomic.h> has been included. Before C++23, this otherwise leads to obscure errors
// because <atomic> may try to redefine things defined by <stdatomic.h>.

// Ignore additional weird errors that happen when the two headers are mixed.
// ADDITIONAL_COMPILE_FLAGS: -Xclang -verify-ignore-unexpected=error -Xclang -verify-ignore-unexpected=warning

#include <stdatomic.h>
#include <atomic>

// expected-error@*:* {{<atomic> is incompatible with <stdatomic.h> before C++23.}}
