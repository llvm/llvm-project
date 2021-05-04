//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <numeric>
// UNSUPPORTED: clang-8
// UNSUPPORTED: gcc-9

// Became constexpr in C++20
// template <InputIterator Iter1, InputIterator Iter2, MoveConstructible T,
//           class BinaryOperation1,
//           Callable<auto, Iter1::reference, Iter2::reference> BinaryOperation2>
//   requires Callable<BinaryOperation1, const T&, BinaryOperation2::result_type>
//         && HasAssign<T, BinaryOperation1::result_type>
//         && CopyConstructible<BinaryOperation1>
//         && CopyConstructible<BinaryOperation2>
//   T
//   inner_product(Iter1 first1, Iter1 last1, Iter2 first2,
//                 T init, BinaryOperation1 binary_op1, BinaryOperation2 binary_op2);

#include <numeric>
#include <functional>
#include <string>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

#if TEST_STD_VER > 17
struct do_nothing_op
{
    template<class T>
    constexpr T operator()(T a, T)
    { return a; }
};

struct rvalue_addable
{
    bool correctOperatorUsed = false;

    constexpr rvalue_addable operator*(rvalue_addable const&) { return *this; }

    // make sure the predicate is passed an rvalue and an lvalue (so check that the first argument was moved)
    constexpr rvalue_addable operator()(rvalue_addable&& r, rvalue_addable const&) {
        r.correctOperatorUsed = true;
        return std::move(r);
    }
};

constexpr rvalue_addable operator+(rvalue_addable& lhs, rvalue_addable const&)
{
    lhs.correctOperatorUsed = false;
    return lhs;
}

constexpr rvalue_addable operator+(rvalue_addable&& lhs, rvalue_addable const&)
{
    lhs.correctOperatorUsed = true;
    return std::move(lhs);
}

constexpr void
test_use_move()
{
    rvalue_addable arr[100];
    auto res1 = std::inner_product(arr, arr + 100, arr, rvalue_addable());
    auto res2 = std::inner_product(arr, arr + 100, arr, rvalue_addable(), /*predicate=*/rvalue_addable(), do_nothing_op());

    assert(res1.correctOperatorUsed);
    assert(res2.correctOperatorUsed);
}
#endif // TEST_STD_VER > 17

// C++20 can use string in constexpr evaluation, but both libc++ and MSVC
// don't have the support yet. In these cases omit the constexpr test.
// FIXME Remove constexpr string workaround introduced in D90569
#if TEST_STD_VER > 17 && \
	(!defined(__cpp_lib_constexpr_string) || __cpp_lib_constexpr_string < 201907L)
void
#else
TEST_CONSTEXPR_CXX20 void
#endif
test_string()
{
    std::string sa[] = {"a", "b", "c"};
    assert(std::accumulate(sa, sa + 3, std::string()) == "abc");
    assert(std::accumulate(sa, sa + 3, std::string(), std::plus<std::string>()) == "abc");
}

template <class Iter1, class Iter2, class T>
TEST_CONSTEXPR_CXX20 void
test(Iter1 first1, Iter1 last1, Iter2 first2, T init, T x)
{
    assert(std::inner_product(first1, last1, first2, init,
           std::multiplies<int>(), std::plus<int>()) == x);
}

template <class Iter1, class Iter2>
TEST_CONSTEXPR_CXX20 void
test()
{
    int a[] = {1, 2, 3, 4, 5, 6};
    int b[] = {6, 5, 4, 3, 2, 1};
    unsigned sa = sizeof(a) / sizeof(a[0]);
    test(Iter1(a), Iter1(a), Iter2(b), 1, 1);
    test(Iter1(a), Iter1(a), Iter2(b), 10, 10);
    test(Iter1(a), Iter1(a+1), Iter2(b), 1, 7);
    test(Iter1(a), Iter1(a+1), Iter2(b), 10, 70);
    test(Iter1(a), Iter1(a+2), Iter2(b), 1, 49);
    test(Iter1(a), Iter1(a+2), Iter2(b), 10, 490);
    test(Iter1(a), Iter1(a+sa), Iter2(b), 1, 117649);
    test(Iter1(a), Iter1(a+sa), Iter2(b), 10, 1176490);
}

TEST_CONSTEXPR_CXX20 bool
test()
{
    test<cpp17_input_iterator<const int*>, cpp17_input_iterator<const int*> >();
    test<cpp17_input_iterator<const int*>, forward_iterator<const int*> >();
    test<cpp17_input_iterator<const int*>, bidirectional_iterator<const int*> >();
    test<cpp17_input_iterator<const int*>, random_access_iterator<const int*> >();
    test<cpp17_input_iterator<const int*>, const int*>();

    test<forward_iterator<const int*>, cpp17_input_iterator<const int*> >();
    test<forward_iterator<const int*>, forward_iterator<const int*> >();
    test<forward_iterator<const int*>, bidirectional_iterator<const int*> >();
    test<forward_iterator<const int*>, random_access_iterator<const int*> >();
    test<forward_iterator<const int*>, const int*>();

    test<bidirectional_iterator<const int*>, cpp17_input_iterator<const int*> >();
    test<bidirectional_iterator<const int*>, forward_iterator<const int*> >();
    test<bidirectional_iterator<const int*>, bidirectional_iterator<const int*> >();
    test<bidirectional_iterator<const int*>, random_access_iterator<const int*> >();
    test<bidirectional_iterator<const int*>, const int*>();

    test<random_access_iterator<const int*>, cpp17_input_iterator<const int*> >();
    test<random_access_iterator<const int*>, forward_iterator<const int*> >();
    test<random_access_iterator<const int*>, bidirectional_iterator<const int*> >();
    test<random_access_iterator<const int*>, random_access_iterator<const int*> >();
    test<random_access_iterator<const int*>, const int*>();

    test<const int*, cpp17_input_iterator<const int*> >();
    test<const int*, forward_iterator<const int*> >();
    test<const int*, bidirectional_iterator<const int*> >();
    test<const int*, random_access_iterator<const int*> >();
    test<const int*, const int*>();

#if TEST_STD_VER > 17
    test_use_move();
#endif // TEST_STD_VER > 17
    // C++20 can use string in constexpr evaluation, but both libc++ and MSVC
    // don't have the support yet. In these cases omit the constexpr test.
    // FIXME Remove constexpr string workaround introduced in D90569
#if TEST_STD_VER > 17 && \
	(!defined(__cpp_lib_constexpr_string) || __cpp_lib_constexpr_string < 201907L)
	if (!std::is_constant_evaluated())
#endif
    test_string();

    return true;
}

int main(int, char**)
{
    test();
#if TEST_STD_VER > 17
    static_assert(test());
#endif
    return 0;
}
