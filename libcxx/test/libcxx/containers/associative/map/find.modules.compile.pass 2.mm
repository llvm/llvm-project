//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure that we don't get a compiler error when trying to use std::map::find
// from Objective-C++. This happened in Objective-C++ mode with modules enabled (rdar://106813461).

// REQUIRES: objective-c++

#include <map>

void f(std::map<int, int> const& map, int key) {
    (void)map.find(key);
}
