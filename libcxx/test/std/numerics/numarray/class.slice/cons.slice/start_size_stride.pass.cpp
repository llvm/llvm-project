//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <valarray>

// class slice;

// slice(size_t start, size_t size, size_t stride);

#include <valarray>
#include <cassert>

int main()
{
    std::slice s(1, 3, 2);
    assert(s.start() == 1);
    assert(s.size() == 3);
    assert(s.stride() == 2);
}
