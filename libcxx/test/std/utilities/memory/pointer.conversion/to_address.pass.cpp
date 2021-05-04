//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template <class T> constexpr T* to_address(T* p) noexcept;
// template <class Ptr> constexpr auto to_address(const Ptr& p) noexcept;

#include <memory>
#include <cassert>
#include "test_macros.h"

struct Irrelevant {};

struct P1 {
    using element_type = Irrelevant;
    constexpr explicit P1(int *p) : p_(p) { }
    constexpr int *operator->() const { return p_; }
    int *p_;
};

struct P2 {
    using element_type = Irrelevant;
    constexpr explicit P2(int *p) : p_(p) { }
    constexpr P1 operator->() const { return p_; }
    P1 p_;
};

struct P3 {
    constexpr explicit P3(int *p) : p_(p) { }
    int *p_;
};

template<>
struct std::pointer_traits<P3> {
    static constexpr int *to_address(const P3& p) { return p.p_; }
};

struct P4 {
    constexpr explicit P4(int *p) : p_(p) { }
    int *operator->() const;  // should never be called
    int *p_;
};

template<>
struct std::pointer_traits<P4> {
    static constexpr int *to_address(const P4& p) { return p.p_; }
};

struct P5 {
    using element_type = Irrelevant;
    int const* const& operator->() const;
};

struct P6 {};

template<>
struct std::pointer_traits<P6> {
    static int const* const& to_address(const P6&);
};

constexpr bool test() {
    int i = 0;
    ASSERT_NOEXCEPT(std::to_address(&i));
    assert(std::to_address(&i) == &i);
    P1 p1(&i);
    ASSERT_NOEXCEPT(std::to_address(p1));
    assert(std::to_address(p1) == &i);
    P2 p2(&i);
    ASSERT_NOEXCEPT(std::to_address(p2));
    assert(std::to_address(p2) == &i);
    P3 p3(&i);
    ASSERT_NOEXCEPT(std::to_address(p3));
    assert(std::to_address(p3) == &i);
    P4 p4(&i);
    ASSERT_NOEXCEPT(std::to_address(p4));
    assert(std::to_address(p4) == &i);

    ASSERT_SAME_TYPE(decltype(std::to_address(std::declval<int const*>())), int const*);
    ASSERT_SAME_TYPE(decltype(std::to_address(std::declval<P5>())), int const*);
    ASSERT_SAME_TYPE(decltype(std::to_address(std::declval<P6>())), int const*);

    return true;
}

int main(int, char**) {
    test();
    static_assert(test());
    return 0;
}
