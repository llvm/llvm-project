//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// move_iterator

// template <class U>
//   requires HasAssign<Iter, const U&>
//   move_iterator&
//   operator=(const move_iterator<U>& u);
//
//  constexpr in C++17

#include <iterator>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

template <class It, class U>
void
test(U u)
{
    const std::move_iterator<U> r2(u);
    std::move_iterator<It> r1;
    std::move_iterator<It>& rr = (r1 = r2);
    assert(r1.base() == u);
    assert(&rr == &r1);
}

struct Base {};
struct Derived : Base {};

struct ToIter {
    typedef std::forward_iterator_tag iterator_category;
    typedef char *pointer;
    typedef char &reference;
    typedef char value_type;
    typedef value_type difference_type;

    explicit TEST_CONSTEXPR_CXX17 ToIter() : m_value(0) {}
    TEST_CONSTEXPR_CXX17 ToIter(const ToIter &src) : m_value(src.m_value) {}
    // Intentionally not defined, must not be called.
    ToIter(char *src);
    TEST_CONSTEXPR_CXX17 ToIter &operator=(char *src) {
        m_value = src;
        return *this;
    }
    TEST_CONSTEXPR_CXX17 ToIter &operator=(const ToIter &src) {
        m_value = src.m_value;
        return *this;
    }
    char *m_value;
};

TEST_CONSTEXPR_CXX17 bool test_conv_assign()
{
    char c = '\0';
    char *fi = &c;
    const std::move_iterator<char *> move_fi(fi);
    std::move_iterator<ToIter> move_ti;
    move_ti = move_fi;
    assert(move_ti.base().m_value == fi);
    return true;
}

int main(int, char**)
{
    Derived d;

    test<cpp17_input_iterator<Base*> >(cpp17_input_iterator<Derived*>(&d));
    test<forward_iterator<Base*> >(forward_iterator<Derived*>(&d));
    test<bidirectional_iterator<Base*> >(bidirectional_iterator<Derived*>(&d));
    test<random_access_iterator<const Base*> >(random_access_iterator<Derived*>(&d));
    test<Base*>(&d);
    test_conv_assign();
#if TEST_STD_VER > 14
    {
    using BaseIter    = std::move_iterator<const Base *>;
    using DerivedIter = std::move_iterator<const Derived *>;
    constexpr const Derived *p = nullptr;
    constexpr DerivedIter     it1 = std::make_move_iterator(p);
    constexpr BaseIter        it2 = (BaseIter{nullptr} = it1);
    static_assert(it2.base() == p, "");
    static_assert(test_conv_assign(), "");
    }
#endif

  return 0;
}
