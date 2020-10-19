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

// Use the first element as a position into the data
extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t *data, std::size_t size) {
    if (size <= 1) return 0;
    const size_t partition_point = data[0] % size;
    std::vector<std::uint8_t> working(data + 1, data + size);
    const auto partition_iter = working.begin() + partition_point;
    std::nth_element(working.begin(), partition_iter, working.end());

    // nth may be the end iterator, in this case nth_element has no effect.
    if (partition_iter == working.end()) {
        if (!std::equal(data + 1, data + size, working.begin()))
            return 98;
    }
    else {
        const std::uint8_t nth = *partition_iter;
        if (!std::all_of(working.begin(), partition_iter, [=](std::uint8_t v) { return v <= nth; }))
            return 1;
        if (!std::all_of(partition_iter, working.end(),   [=](std::uint8_t v) { return v >= nth; }))
            return 2;
        if (!fast_is_permutation(data + 1, data + size, working.cbegin()))
            return 99;
    }

    return 0;
}
