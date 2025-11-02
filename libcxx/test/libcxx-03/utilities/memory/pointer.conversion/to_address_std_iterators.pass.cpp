//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class T> constexpr T* __to_address(T* p) noexcept;
// template <class Ptr> constexpr auto __to_address(const Ptr& p) noexcept;

#include <memory>

#include <array>
#include <cassert>
#include <span>
#include <string>
#include <string_view>
#include <valarray>
#include <vector>
#include "test_macros.h"

template<class C>
void test_container_iterators(C c)
{
    const C& cc = c;
    assert(std::__to_address(c.begin()) == c.data());
    assert(std::__to_address(c.end()) == c.data() + c.size());
    assert(std::__to_address(cc.begin()) == cc.data());
    assert(std::__to_address(cc.end()) == cc.data() + cc.size());
}

void test_valarray_iterators()
{
    std::valarray<int> v(100);
    int *p = std::__to_address(std::begin(v));
    int *q = std::__to_address(std::end(v));
    assert(q - p == 100);
}

int main(int, char**) {
    test_container_iterators(std::array<int, 3>());
    test_container_iterators(std::vector<int>(3));
    test_container_iterators(std::string("abc"));
#if TEST_STD_VER >= 17
    test_container_iterators(std::string_view("abc"));
#endif
#if TEST_STD_VER >= 20
    test_container_iterators(std::span<const char>("abc"));
#endif
    test_valarray_iterators();

    return 0;
}
