//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// std::iterator

#include <iterator>

std::iterator<std::input_iterator_tag, char> it; // expected-warning-re {{'iterator<{{.+}}>' is deprecated}}
