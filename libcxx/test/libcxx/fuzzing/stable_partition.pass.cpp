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
    auto is_even = [](auto b) { return b.key % 2 == 0; };

    std::vector<ByteWithPayload> input;
    for (std::size_t i = 0; i < size; ++i)
        input.push_back(ByteWithPayload(data[i], i));
    std::vector<ByteWithPayload> working = input;
    auto iter = std::stable_partition(working.begin(), working.end(), is_even);

    if (!std::all_of(working.begin(), iter, is_even))
        return 1;
    if (!std::none_of(iter,   working.end(), is_even))
        return 2;
    if (!std::is_sorted(working.begin(), iter, ByteWithPayload::payload_less()))
        return 3;
    if (!std::is_sorted(iter,   working.end(), ByteWithPayload::payload_less()))
        return 4;
    if (!fast_is_permutation(input.cbegin(), input.cend(), working.cbegin()))
        return 99;
    return 0;
}
