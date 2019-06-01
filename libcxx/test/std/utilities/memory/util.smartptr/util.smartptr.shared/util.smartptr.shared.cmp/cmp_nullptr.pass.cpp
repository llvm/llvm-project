//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// shared_ptr

// template <class T>
//     bool operator==(const shared_ptr<T>& x, nullptr_t) noexcept;
// template <class T>
//     bool operator==(nullptr_t, const shared_ptr<T>& y) noexcept;
// template <class T>
//     bool operator!=(const shared_ptr<T>& x, nullptr_t) noexcept;
// template <class T>
//     bool operator!=(nullptr_t, const shared_ptr<T>& y) noexcept;
// template <class T>
//     bool operator<(const shared_ptr<T>& x, nullptr_t) noexcept;
// template <class T>
//     bool operator<(nullptr_t, const shared_ptr<T>& y) noexcept;
// template <class T>
//     bool operator<=(const shared_ptr<T>& x, nullptr_t) noexcept;
// template <class T>
//     bool operator<=(nullptr_t, const shared_ptr<T>& y) noexcept;
// template <class T>
//     bool operator>(const shared_ptr<T>& x, nullptr_t) noexcept;
// template <class T>
//     bool operator>(nullptr_t, const shared_ptr<T>& y) noexcept;
// template <class T>
//     bool operator>=(const shared_ptr<T>& x, nullptr_t) noexcept;
// template <class T>
//     bool operator>=(nullptr_t, const shared_ptr<T>& y) noexcept;

#include <memory>
#include <cassert>

#include "test_macros.h"

void do_nothing(int*) {}

int main(int, char**)
{
    const std::shared_ptr<int> p1(new int(1));
    assert(!(p1 == nullptr));
    assert(!(nullptr == p1));
    assert(!(p1 < nullptr));
    assert( (nullptr < p1));
    assert(!(p1 <= nullptr));
    assert( (nullptr <= p1));
    assert( (p1 > nullptr));
    assert(!(nullptr > p1));
    assert( (p1 >= nullptr));
    assert(!(nullptr >= p1));

    const std::shared_ptr<int> p2;
    assert( (p2 == nullptr));
    assert( (nullptr == p2));
    assert(!(p2 < nullptr));
    assert(!(nullptr < p2));
    assert( (p2 <= nullptr));
    assert( (nullptr <= p2));
    assert(!(p2 > nullptr));
    assert(!(nullptr > p2));
    assert( (p2 >= nullptr));
    assert( (nullptr >= p2));

  return 0;
}
