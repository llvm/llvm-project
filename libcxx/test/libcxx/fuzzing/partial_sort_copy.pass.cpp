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

// Use the first element as a count
extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t *data, std::size_t size) {
    if (size <= 1)
        return 0;
    const std::size_t num_results = data[0] % size;
    std::vector<std::uint8_t> results(num_results);
    (void)std::partial_sort_copy(data + 1, data + size, results.begin(), results.end());

    // The results have to be sorted
    if (!std::is_sorted(results.begin(), results.end()))
        return 1;
    // All the values in results have to be in the original data
    for (auto v: results)
        if (std::find(data + 1, data + size, v) == data + size)
            return 2;

    // The things in results have to be the smallest N in the original data
    std::vector<std::uint8_t> sorted(data + 1, data + size);
    std::sort(sorted.begin(), sorted.end());
    if (!std::equal(results.begin(), results.end(), sorted.begin()))
        return 3;

    return 0;
}
