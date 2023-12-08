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
    auto is_even = [](auto x) { return x % 2 == 0; };
    std::vector<std::uint8_t> working(data, data + size);
    auto iter = std::partition(working.begin(), working.end(), is_even);

    if (!std::all_of(working.begin(), iter, is_even))
        return 1;
    if (!std::none_of(iter,   working.end(), is_even))
        return 2;
    if (!fast_is_permutation(data, data + size, working.cbegin()))
        return 99;
    return 0;
}
