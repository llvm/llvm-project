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
#include <iterator>
#include <vector>

#include "fuzz.h"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t *data, std::size_t size) {
    std::vector<std::uint8_t> working(data, data + size);
    std::sort(working.begin(), working.end());
    std::vector<std::uint8_t> results;
    (void)std::unique_copy(working.begin(), working.end(),
                           std::back_inserter<std::vector<std::uint8_t>>(results));
    std::vector<std::uint8_t>::iterator it; // scratch iterator

    // Check the size of the unique'd sequence.
    // it should only be zero if the input sequence was empty.
    if (results.size() == 0)
        return working.size() == 0 ? 0 : 1;

    // 'results' is sorted
    if (!std::is_sorted(results.begin(), results.end()))
        return 2;

    // All the elements in 'results' must be different
    it = results.begin();
    std::uint8_t prev_value = *it++;
    for (; it != results.end(); ++it) {
        if (*it == prev_value)
            return 3;
        prev_value = *it;
    }

    // Every element in 'results' must be in 'working'
    for (auto v : results)
        if (std::find(working.begin(), working.end(), v) == working.end())
            return 4;

    // Every element in 'working' must be in 'results'
    for (auto v : working)
        if (std::find(results.begin(), results.end(), v) == results.end())
            return 5;

    return 0;
}
