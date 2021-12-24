//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//  A set of routines for testing the comparison operators of a type
//
//      XXXX6 tests all six comparison operators
//      XXXX2 tests only op== and op!=
//
//      AssertComparisonsXAreNoexcept       static_asserts that the operations are all noexcept.
//      AssertComparisonsXReturnBool        static_asserts that the operations return bool.
//      AssertComparisonsXConvertibleToBool static_asserts that the operations return something convertible to bool.


#ifndef TEST_COMPARISONS_H
#define TEST_COMPARISONS_H

#include <type_traits>
#include <cassert>
#include "test_macros.h"

//  Test all six comparison operations for sanity
template <class T, class U = T>
TEST_CONSTEXPR_CXX14 bool testComparisons6(const T& t1, const U& t2, bool isEqual, bool isLess)
{
    assert(!(isEqual && isLess) && "isEqual and isLess cannot be both true");
    if (isEqual)
        {
        if (!(t1 == t2)) return false;
        if (!(t2 == t1)) return false;
        if ( (t1 != t2)) return false;
        if ( (t2 != t1)) return false;
        if ( (t1  < t2)) return false;
        if ( (t2  < t1)) return false;
        if (!(t1 <= t2)) return false;
        if (!(t2 <= t1)) return false;
        if ( (t1  > t2)) return false;
        if ( (t2  > t1)) return false;
        if (!(t1 >= t2)) return false;
        if (!(t2 >= t1)) return false;
        }
    else if (isLess)
        {
        if ( (t1 == t2)) return false;
        if ( (t2 == t1)) return false;
        if (!(t1 != t2)) return false;
        if (!(t2 != t1)) return false;
        if (!(t1  < t2)) return false;
        if ( (t2  < t1)) return false;
        if (!(t1 <= t2)) return false;
        if ( (t2 <= t1)) return false;
        if ( (t1  > t2)) return false;
        if (!(t2  > t1)) return false;
        if ( (t1 >= t2)) return false;
        if (!(t2 >= t1)) return false;
        }
    else /* greater */
        {
        if ( (t1 == t2)) return false;
        if ( (t2 == t1)) return false;
        if (!(t1 != t2)) return false;
        if (!(t2 != t1)) return false;
        if ( (t1  < t2)) return false;
        if (!(t2  < t1)) return false;
        if ( (t1 <= t2)) return false;
        if (!(t2 <= t1)) return false;
        if (!(t1  > t2)) return false;
        if ( (t2  > t1)) return false;
        if (!(t1 >= t2)) return false;
        if ( (t2 >= t1)) return false;
        }

    return true;
}

//  Easy call when you can init from something already comparable.
template <class T, class Param>
TEST_CONSTEXPR_CXX14 bool testComparisons6Values(Param val1, Param val2)
{
    const bool isEqual = val1 == val2;
    const bool isLess  = val1  < val2;

    return testComparisons6(T(val1), T(val2), isEqual, isLess);
}

template <class T, class U = T>
void AssertComparisons6AreNoexcept()
{
    ASSERT_NOEXCEPT(std::declval<const T&>() == std::declval<const U&>());
    ASSERT_NOEXCEPT(std::declval<const T&>() != std::declval<const U&>());
    ASSERT_NOEXCEPT(std::declval<const T&>() <  std::declval<const U&>());
    ASSERT_NOEXCEPT(std::declval<const T&>() <= std::declval<const U&>());
    ASSERT_NOEXCEPT(std::declval<const T&>() >  std::declval<const U&>());
    ASSERT_NOEXCEPT(std::declval<const T&>() >= std::declval<const U&>());
}

template <class T, class U = T>
void AssertComparisons6ReturnBool()
{
    ASSERT_SAME_TYPE(decltype(std::declval<const T&>() == std::declval<const U&>()), bool);
    ASSERT_SAME_TYPE(decltype(std::declval<const T&>() != std::declval<const U&>()), bool);
    ASSERT_SAME_TYPE(decltype(std::declval<const T&>() <  std::declval<const U&>()), bool);
    ASSERT_SAME_TYPE(decltype(std::declval<const T&>() <= std::declval<const U&>()), bool);
    ASSERT_SAME_TYPE(decltype(std::declval<const T&>() >  std::declval<const U&>()), bool);
    ASSERT_SAME_TYPE(decltype(std::declval<const T&>() >= std::declval<const U&>()), bool);
}


template <class T, class U = T>
void AssertComparisons6ConvertibleToBool()
{
    static_assert((std::is_convertible<decltype(std::declval<const T&>() == std::declval<const U&>()), bool>::value), "");
    static_assert((std::is_convertible<decltype(std::declval<const T&>() != std::declval<const U&>()), bool>::value), "");
    static_assert((std::is_convertible<decltype(std::declval<const T&>() <  std::declval<const U&>()), bool>::value), "");
    static_assert((std::is_convertible<decltype(std::declval<const T&>() <= std::declval<const U&>()), bool>::value), "");
    static_assert((std::is_convertible<decltype(std::declval<const T&>() >  std::declval<const U&>()), bool>::value), "");
    static_assert((std::is_convertible<decltype(std::declval<const T&>() >= std::declval<const U&>()), bool>::value), "");
}

//  Test all two comparison operations for sanity
template <class T, class U = T>
TEST_CONSTEXPR_CXX14 bool testComparisons2(const T& t1, const U& t2, bool isEqual)
{
    if (isEqual)
        {
        if (!(t1 == t2)) return false;
        if (!(t2 == t1)) return false;
        if ( (t1 != t2)) return false;
        if ( (t2 != t1)) return false;
        }
    else /* not equal */
        {
        if ( (t1 == t2)) return false;
        if ( (t2 == t1)) return false;
        if (!(t1 != t2)) return false;
        if (!(t2 != t1)) return false;
        }

    return true;
}

//  Easy call when you can init from something already comparable.
template <class T, class Param>
TEST_CONSTEXPR_CXX14 bool testComparisons2Values(Param val1, Param val2)
{
    const bool isEqual = val1 == val2;

    return testComparisons2(T(val1), T(val2), isEqual);
}

template <class T, class U = T>
void AssertComparisons2AreNoexcept()
{
    ASSERT_NOEXCEPT(std::declval<const T&>() == std::declval<const U&>());
    ASSERT_NOEXCEPT(std::declval<const T&>() != std::declval<const U&>());
}

template <class T, class U = T>
void AssertComparisons2ReturnBool()
{
    ASSERT_SAME_TYPE(decltype(std::declval<const T&>() == std::declval<const U&>()), bool);
    ASSERT_SAME_TYPE(decltype(std::declval<const T&>() != std::declval<const U&>()), bool);
}


template <class T, class U = T>
void AssertComparisons2ConvertibleToBool()
{
    static_assert((std::is_convertible<decltype(std::declval<const T&>() == std::declval<const U&>()), bool>::value), "");
    static_assert((std::is_convertible<decltype(std::declval<const T&>() != std::declval<const U&>()), bool>::value), "");
}

struct LessAndEqComp {
  int value;

  TEST_CONSTEXPR_CXX14 LessAndEqComp(int v) : value(v) {}

  friend TEST_CONSTEXPR_CXX14 bool operator<(const LessAndEqComp& lhs, const LessAndEqComp& rhs) {
    return lhs.value < rhs.value;
  }

  friend TEST_CONSTEXPR_CXX14 bool operator==(const LessAndEqComp& lhs, const LessAndEqComp& rhs) {
    return lhs.value == rhs.value;
  }
};
#endif // TEST_COMPARISONS_H
