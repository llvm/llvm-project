//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: c++17

// std::raw_storage_iterator

#include <memory>

std::raw_storage_iterator<int*, int> it(nullptr);
// expected-warning@-1{{'raw_storage_iterator<int *, int>' is deprecated}}
