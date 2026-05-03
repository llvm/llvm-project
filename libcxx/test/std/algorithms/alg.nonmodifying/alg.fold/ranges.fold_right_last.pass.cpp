//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===

#include <algorithm>
#include <cassert>
#include <concepts>
#include <deque>
#include <forward_list>
#include <functional>
#include <iterator>
#include <list>
#include <optional>
#include <ranges>
#include <set>
#include <string_view>
#include <string>
#include <vector>

#include "test_macros.h"
#include "test_range.h"
#include "invocable_with_telemetry.h"
#include "maths.h"

constexpr bool test() {
    auto op = [](int a, int b) { return a + b; };
    {
        std::vector<int> v = {1, 2, 3};
        auto res = std::ranges::fold_right_last(v, op);
        assert(res.has_value());
        assert(res.value() == 6); 
    }

    {
        std::vector<int> empty = {};
        auto res = std::ranges::fold_right_last(empty, op);
        assert(!res.has_value());
    }

    return true;
}

int main(int, char**) {
    test();
    static_assert(test());
    return 0;
}
