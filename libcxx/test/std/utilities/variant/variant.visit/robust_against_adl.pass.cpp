//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <variant>
// template <class Visitor, class... Variants>
// constexpr see below visit(Visitor&& vis, Variants&&... vars);

#include <variant>

#include "test_macros.h"

struct Incomplete;
template<class T> struct Holder { T t; };

constexpr bool test(bool do_it)
{
    if (do_it) {
        std::variant<Holder<Incomplete>*, int> v = nullptr;
        std::visit([](auto){}, v);
        std::visit([](auto) -> Holder<Incomplete>* { return nullptr; }, v);
#if TEST_STD_VER > 17
        std::visit<void>([](auto){}, v);
        std::visit<void*>([](auto) -> Holder<Incomplete>* { return nullptr; }, v);
#endif
    }
    return true;
}

int main(int, char**)
{
    test(true);
#if TEST_STD_VER > 17
    static_assert(test(true));
#endif
    return 0;
}
