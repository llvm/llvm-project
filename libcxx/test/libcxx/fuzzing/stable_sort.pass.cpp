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
    std::vector<ByteWithPayload> input;
    for (std::size_t i = 0; i < size; ++i)
        input.push_back(ByteWithPayload(data[i], i));

    std::vector<ByteWithPayload> working = input;
    std::stable_sort(working.begin(), working.end(), ByteWithPayload::key_less());

    if (!std::is_sorted(working.begin(), working.end(), ByteWithPayload::key_less()))
        return 1;

    auto iter = working.begin();
    while (iter != working.end()) {
        auto range = std::equal_range(iter, working.end(), *iter, ByteWithPayload::key_less());
        if (!std::is_sorted(range.first, range.second, ByteWithPayload::total_less()))
            return 2;
        iter = range.second;
    }
    if (!fast_is_permutation(input.cbegin(), input.cend(), working.cbegin()))
        return 99;
    return 0;
}
