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
    auto is_even = [](auto t) {
        return t % 2 == 0;
    };

    std::vector<std::uint8_t> v1, v2;
    auto iter = std::partition_copy(data, data + size,
        std::back_inserter<std::vector<std::uint8_t>>(v1),
        std::back_inserter<std::vector<std::uint8_t>>(v2),
        is_even);
    ((void)iter);
    // The two vectors should add up to the original size
    if (v1.size() + v2.size() != size)
        return 1;

    // All of the even values should be in the first vector, and none in the second
    if (!std::all_of(v1.begin(), v1.end(), is_even))
        return 2;
    if (!std::none_of(v2.begin(), v2.end(), is_even))
        return 3;

    // Every value in both vectors has to be in the original

    // Make a copy of the input, and sort it
    std::vector<std::uint8_t> v0{data, data + size};
    std::sort(v0.begin(), v0.end());

    // Sort each vector and ensure that all of the elements appear in the original input
    std::sort(v1.begin(), v1.end());
    if (!std::includes(v0.begin(), v0.end(), v1.begin(), v1.end()))
        return 4;

    std::sort(v2.begin(), v2.end());
    if (!std::includes(v0.begin(), v0.end(), v2.begin(), v2.end()))
        return 5;

    // This, while simple, is really slow - 20 seconds on a 500K element input.
    //  for (auto v: v1)
    //      if (std::find(data, data + size, v) == data + size)
    //          return 4;
    //
    //  for (auto v: v2)
    //      if (std::find(data, data + size, v) == data + size)
    //          return 5;

    return 0;
}
