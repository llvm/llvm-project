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
    std::vector<std::uint8_t> working(data, data + size);
    std::make_heap(working.begin(), working.end());

    // Pop things off, one at a time
    auto iter = --working.end();
    while (iter != working.begin()) {
        std::pop_heap(working.begin(), iter);
        if (!std::is_heap(working.begin(), --iter))
            return 2;
    }

    return 0;
}
