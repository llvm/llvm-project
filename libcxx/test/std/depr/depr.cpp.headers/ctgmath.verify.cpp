//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ctgmath>

// check that <ctgmath> is deprecated in C++17 and removed in C++20
// When built with modules, <ctgmath> should be omitted.

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: clang-modules-build

// FIXME: using `#warning` causes diagnostics from system headers which include deprecated headers. This can only be
// enabled again once https://github.com/llvm/llvm-project/pull/168041 (or a similar feature) has landed, since that
// allows suppression in system headers.
// XFAIL: *

#include <ctgmath>

// expected-warning@ctgmath:* {{<ctgmath> is deprecated in C++17 and removed in C++20. Include <cmath> and <complex> instead.}}
