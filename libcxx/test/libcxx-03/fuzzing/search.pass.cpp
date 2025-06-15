//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "fuzz.h"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t *data, std::size_t size) {
    if (size < 2)
        return 0;

    const std::size_t pat_size = data[0] * (size - 1) / std::numeric_limits<uint8_t>::max();
    assert(pat_size <= size - 1);
    const std::uint8_t *pat_begin = data + 1;
    const std::uint8_t *pat_end   = pat_begin + pat_size;
    const std::uint8_t *data_end  = data + size;
    assert(pat_end <= data_end);

    auto it = std::search(pat_end, data_end, pat_begin, pat_end);
    if (it != data_end) // not found
        if (!std::equal(pat_begin, pat_end, it))
            return 1;
    return 0;
}
