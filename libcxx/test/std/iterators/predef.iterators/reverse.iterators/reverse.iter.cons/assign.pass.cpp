//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// reverse_iterator

// template <class U>
// reverse_iterator& operator=(const reverse_iterator<U>& u); // constexpr since C++17

#include <iterator>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

template <class It, class U>
TEST_CONSTEXPR_CXX17 void test(U u) {
    const std::reverse_iterator<U> r2(u);
    std::reverse_iterator<It> r1;
    std::reverse_iterator<It>& rr = r1 = r2;
    assert(r1.base() == u);
    assert(&rr == &r1);
}

struct Base { };
struct Derived : Base { };

struct ToIter {
    typedef std::bidirectional_iterator_tag iterator_category;
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

TEST_CONSTEXPR_CXX17 bool tests() {
    Derived d;
    test<bidirectional_iterator<Base*> >(bidirectional_iterator<Derived*>(&d));
    test<random_access_iterator<const Base*> >(random_access_iterator<Derived*>(&d));
    test<Base*>(&d);

    char c = '\0';
    char *fi = &c;
    const std::reverse_iterator<char *> rev_fi(fi);
    std::reverse_iterator<ToIter> rev_ti;
    rev_ti = rev_fi;
    assert(rev_ti.base().m_value == fi);

    return true;
}

int main(int, char**) {
    tests();
#if TEST_STD_VER > 14
    static_assert(tests(), "");
#endif
    return 0;
}
