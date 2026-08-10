//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "fuzz.h"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t *data, std::size_t size) {
    if (size < 2)
        return 0;

    // Make a heap from the first half of the data
    std::vector<std::uint8_t> working(data, data + size);
    auto iter = working.begin() + (size / 2);
    std::make_heap(working.begin(), iter);
    if (!std::is_heap(working.begin(), iter))
        return 1;

    // Now push the rest onto the heap, one at a time
    ++iter;
    for (; iter != working.end(); ++iter) {
        std::push_heap(working.begin(), iter);
        if (!std::is_heap(working.begin(), iter))
            return 2;
    }

    if (!fast_is_permutation(data, data + size, working.cbegin()))
        return 99;
    return 0;
}
