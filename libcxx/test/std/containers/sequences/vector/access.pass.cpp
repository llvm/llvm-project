//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

//       reference operator[](size_type __i);
// const_reference operator[](size_type __i) const;
//
//       reference at(size_type __i);
// const_reference at(size_type __i) const;
//
//       reference front();
// const_reference front() const;
//
//       reference back();
// const_reference back() const;
// libc++ marks these as 'noexcept' (except 'at')

#include <vector>
#include <cassert>
#include <stdexcept>

#include "min_allocator.h"
#include "test_macros.h"

template <class C>
C
make(int size, int start)
{
    C c;
    for (int i = 0; i < size; ++i)
        c.push_back(start + i);
    return c;
}

template <class Vector>
void test_get_basic(Vector& c, int start_value) {
    const int n = static_cast<int>(c.size());
    for (int i = 0; i < n; ++i)
        assert(c[i] == start_value + i);
    for (int i = 0; i < n; ++i)
        assert(c.at(i) == start_value + i);

#ifndef TEST_HAS_NO_EXCEPTIONS
    try {
        TEST_IGNORE_NODISCARD c.at(n);
        assert(false);
    } catch (const std::out_of_range&) {}
#endif

    assert(c.front() == start_value);
    assert(c.back() == start_value + n - 1);
}

template <class Vector>
void test_get() {
    int start_value = 35;
    Vector c = make<Vector>(10, start_value);
    const Vector& cc = c;
    test_get_basic(c, start_value);
    test_get_basic(cc, start_value);
}

template <class Vector>
void test_set() {
    int start_value = 35;
    const int n = 10;
    Vector c = make<Vector>(n, start_value);

    for (int i = 0; i < n; ++i) {
        assert(c[i] == start_value + i);
        c[i] = start_value + i + 1;
        assert(c[i] == start_value + i + 1);
    }
    for (int i = 0; i < n; ++i) {
        assert(c.at(i) == start_value + i + 1);
        c.at(i) = start_value + i + 2;
        assert(c.at(i) == start_value + i + 2);
    }

    assert(c.front() == start_value + 2);
    c.front() = start_value + 3;
    assert(c.front() == start_value + 3);

    assert(c.back() == start_value + n + 1);
    c.back() = start_value + n + 2;
    assert(c.back() == start_value + n + 2);
}

template <class Vector>
void test() {
    test_get<Vector>();
    test_set<Vector>();

    Vector c;
    const Vector& cc = c;
    ASSERT_SAME_TYPE(typename Vector::reference, decltype(c[0]));
    ASSERT_SAME_TYPE(typename Vector::const_reference, decltype(cc[0]));

    ASSERT_SAME_TYPE(typename Vector::reference, decltype(c.at(0)));
    ASSERT_SAME_TYPE(typename Vector::const_reference, decltype(cc.at(0)));

    ASSERT_SAME_TYPE(typename Vector::reference, decltype(c.front()));
    ASSERT_SAME_TYPE(typename Vector::const_reference, decltype(cc.front()));

    ASSERT_SAME_TYPE(typename Vector::reference, decltype(c.back()));
    ASSERT_SAME_TYPE(typename Vector::const_reference, decltype(cc.back()));
}

int main(int, char**)
{
    test<std::vector<int> >();
#if TEST_STD_VER >= 11
    test<std::vector<int, min_allocator<int> > >();
#endif

  return 0;
}
