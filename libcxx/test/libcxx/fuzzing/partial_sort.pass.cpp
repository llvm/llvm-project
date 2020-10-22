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
    if (size <= 1)
        return 0;
    const std::size_t sort_point = data[0] % size;
    std::vector<std::uint8_t> working(data + 1, data + size);
    const auto sort_iter = working.begin() + sort_point;
    std::partial_sort(working.begin(), sort_iter, working.end());

    if (sort_iter != working.end()) {
        const std::uint8_t nth = *std::min_element(sort_iter, working.end());
        if (!std::all_of(working.begin(), sort_iter, [=](std::uint8_t v) { return v <= nth; }))
            return 1;
        if (!std::all_of(sort_iter, working.end(),   [=](std::uint8_t v) { return v >= nth; }))
            return 2;
    }
    if (!std::is_sorted(working.begin(), sort_iter))
        return 3;
    if (!fast_is_permutation(data + 1, data + size, working.cbegin()))
        return 99;

    return 0;
}
