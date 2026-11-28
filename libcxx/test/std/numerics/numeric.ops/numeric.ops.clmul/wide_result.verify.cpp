//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++29

// <numeric>
// [numeric.clmul]

#include <numeric>

// expected-error-re@*:* {{static assertion failed{{.*}}}}
// expected-note@+1 {{in instantiation of template class 'std::wide_result<int>' requested here}}
std::wide_result<int> invalid_type(0, 0);
