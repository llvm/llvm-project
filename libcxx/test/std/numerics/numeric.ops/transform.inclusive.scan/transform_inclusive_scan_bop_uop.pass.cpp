//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <numeric>
// UNSUPPORTED: c++98, c++03, c++11, c++14

// template<class InputIterator, class OutputIterator, class T,
//          class BinaryOperation, class UnaryOperation>
//   OutputIterator transform_inclusive_scan(InputIterator first, InputIterator last,
//                                           OutputIterator result,
//                                           BinaryOperation binary_op,
//                                           UnaryOperation unary_op);


#include <numeric>
#include <vector>
#include <cassert>
#include <iostream>

#include "test_iterators.h"

template <class T = void>
struct identity : std::unary_function<T, T>
{
    constexpr const T& operator()(const T& __x) const { return __x;}
};

template <>
struct identity<void>
{
    template <class T>
    constexpr auto operator()(T&& __x) const
    _NOEXCEPT_(noexcept(_VSTD::forward<T>(__x)))
    -> decltype        (_VSTD::forward<T>(__x))
        { return        _VSTD::forward<T>(__x); }
};

template <class Iter1, class BOp, class UOp, class Iter2>
void
test(Iter1 first, Iter1 last, BOp bop, UOp uop, Iter2 rFirst, Iter2 rLast)
{
    std::vector<typename std::iterator_traits<Iter1>::value_type> v;
//  Test not in-place
    std::transform_inclusive_scan(first, last, std::back_inserter(v), bop, uop);
    assert(std::equal(v.begin(), v.end(), rFirst, rLast));

//  Test in-place
    v.clear();
    v.assign(first, last);
    std::transform_inclusive_scan(v.begin(), v.end(), v.begin(), bop, uop);
    assert(std::equal(v.begin(), v.end(), rFirst, rLast));
}


template <class Iter>
void
test()
{
          int ia[]     = {  1,  3,   5,   7,    9};
    const int pResI0[] = {  1,  4,   9,  16,   25};        // with identity
    const int mResI0[] = {  1,  3,  15, 105,  945};
    const int pResN0[] = { -1, -4,  -9, -16,  -25};        // with negate
    const int mResN0[] = { -1,  3, -15, 105, -945};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    static_assert(sa == sizeof(pResI0) / sizeof(pResI0[0]));       // just to be sure
    static_assert(sa == sizeof(mResI0) / sizeof(mResI0[0]));       // just to be sure
    static_assert(sa == sizeof(pResN0) / sizeof(pResN0[0]));       // just to be sure
    static_assert(sa == sizeof(mResN0) / sizeof(mResN0[0]));       // just to be sure

    for (unsigned int i = 0; i < sa; ++i ) {
        test(Iter(ia), Iter(ia + i), std::plus<>(),       identity<>(),    pResI0, pResI0 + i);
        test(Iter(ia), Iter(ia + i), std::multiplies<>(), identity<>(),    mResI0, mResI0 + i);
        test(Iter(ia), Iter(ia + i), std::plus<>(),       std::negate<>(), pResN0, pResN0 + i);
        test(Iter(ia), Iter(ia + i), std::multiplies<>(), std::negate<>(), mResN0, mResN0 + i);
        }
}

int triangle(int n) { return n*(n+1)/2; }

//  Basic sanity
void basic_tests()
{
    {
    std::vector<int> v(10);
    std::fill(v.begin(), v.end(), 3);
    std::transform_inclusive_scan(v.begin(), v.end(), v.begin(), std::plus<>(), identity<>());
    std::copy(v.begin(), v.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
    for (size_t i = 0; i < v.size(); ++i)
        assert(v[i] == (int)(i+1) * 3);
    }

    {
    std::vector<int> v(10);
    std::iota(v.begin(), v.end(), 0);
    std::transform_inclusive_scan(v.begin(), v.end(), v.begin(), std::plus<>(), identity<>());
    for (size_t i = 0; i < v.size(); ++i)
        assert(v[i] == triangle(i));
    }

    {
    std::vector<int> v(10);
    std::iota(v.begin(), v.end(), 1);
    std::transform_inclusive_scan(v.begin(), v.end(), v.begin(), std::plus<>(), identity<>());
    for (size_t i = 0; i < v.size(); ++i)
        assert(v[i] == triangle(i + 1));
    }

    {
    std::vector<int> v, res;
    std::transform_inclusive_scan(v.begin(), v.end(), std::back_inserter(res), std::plus<>(), identity<>());
    assert(res.empty());
    }
}

int main()
{
    basic_tests();

//  All the iterator categories
    test<input_iterator        <const int*> >();
    test<forward_iterator      <const int*> >();
    test<bidirectional_iterator<const int*> >();
    test<random_access_iterator<const int*> >();
    test<const int*>();
    test<      int*>();
}
