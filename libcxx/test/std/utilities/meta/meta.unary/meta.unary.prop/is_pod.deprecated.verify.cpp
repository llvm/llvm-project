//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// <type_traits>

// is_pod and is_pod_v are deprecated in C++20 by P0767R1

#include <type_traits>

static_assert(std::is_pod<int>::value); // expected-warning {{'is_pod<int>' is deprecated}}
static_assert(std::is_pod_v<int>);      // expected-warning {{'is_pod_v<int>' is deprecated}}
