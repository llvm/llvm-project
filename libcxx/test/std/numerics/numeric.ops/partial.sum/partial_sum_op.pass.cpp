//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <numeric>

// Became constexpr in C++20
// template<InputIterator InIter,
//          OutputIterator<auto, const InIter::value_type&> OutIter,
//          Callable<auto, const InIter::value_type&, InIter::reference> BinaryOperation>
//   requires HasAssign<InIter::value_type, BinaryOperation::result_type>
//         && Constructible<InIter::value_type, InIter::reference>
//         && CopyConstructible<BinaryOperation>
//   OutIter
//   partial_sum(InIter first, InIter last, OutIter result, BinaryOperation binary_op);

#include <numeric>
#include <functional>
#include <string>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

#if TEST_STD_VER > 17
struct rvalue_addable
{
    bool correctOperatorUsed = false;

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
    const std::size_t size = 100;
    rvalue_addable arr[size];
    rvalue_addable res1[size];
    rvalue_addable res2[size];
    std::partial_sum(arr, arr + size, res1);
    std::partial_sum(arr, arr + size, res2, /*predicate=*/rvalue_addable());
    // start at 1 because the first element is not moved
    for (unsigned i = 1; i < size; ++i) assert(res1[i].correctOperatorUsed);
    for (unsigned i = 1; i < size; ++i) assert(res2[i].correctOperatorUsed);
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
    std::string sr[] = {"a", "ba", "cb"};
    std::string output[3];
    std::adjacent_difference(sa, sa + 3, output, std::plus<std::string>());
    for (unsigned i = 0; i < 3; ++i) assert(output[i] == sr[i]);
}

template <class InIter, class OutIter>
TEST_CONSTEXPR_CXX20 void
test()
{
    int ia[] = {1, 2, 3, 4, 5};
    int ir[] = {1, -1, -4, -8, -13};
    const unsigned s = sizeof(ia) / sizeof(ia[0]);
    int ib[s] = {0};
    OutIter r = std::partial_sum(InIter(ia), InIter(ia+s), OutIter(ib), std::minus<int>());
    assert(base(r) == ib + s);
    for (unsigned i = 0; i < s; ++i)
        assert(ib[i] == ir[i]);
}

TEST_CONSTEXPR_CXX20 bool
test()
{
    test<cpp17_input_iterator<const int*>, cpp17_output_iterator<int*> >();
    test<cpp17_input_iterator<const int*>, forward_iterator<int*> >();
    test<cpp17_input_iterator<const int*>, bidirectional_iterator<int*> >();
    test<cpp17_input_iterator<const int*>, random_access_iterator<int*> >();
    test<cpp17_input_iterator<const int*>, int*>();

    test<forward_iterator<const int*>, cpp17_output_iterator<int*> >();
    test<forward_iterator<const int*>, forward_iterator<int*> >();
    test<forward_iterator<const int*>, bidirectional_iterator<int*> >();
    test<forward_iterator<const int*>, random_access_iterator<int*> >();
    test<forward_iterator<const int*>, int*>();

    test<bidirectional_iterator<const int*>, cpp17_output_iterator<int*> >();
    test<bidirectional_iterator<const int*>, forward_iterator<int*> >();
    test<bidirectional_iterator<const int*>, bidirectional_iterator<int*> >();
    test<bidirectional_iterator<const int*>, random_access_iterator<int*> >();
    test<bidirectional_iterator<const int*>, int*>();

    test<random_access_iterator<const int*>, cpp17_output_iterator<int*> >();
    test<random_access_iterator<const int*>, forward_iterator<int*> >();
    test<random_access_iterator<const int*>, bidirectional_iterator<int*> >();
    test<random_access_iterator<const int*>, random_access_iterator<int*> >();
    test<random_access_iterator<const int*>, int*>();

    test<const int*, cpp17_output_iterator<int*> >();
    test<const int*, forward_iterator<int*> >();
    test<const int*, bidirectional_iterator<int*> >();
    test<const int*, random_access_iterator<int*> >();
    test<const int*, int*>();

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
