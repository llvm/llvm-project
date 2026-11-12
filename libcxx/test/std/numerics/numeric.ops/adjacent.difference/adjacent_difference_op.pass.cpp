//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <numeric>

// Became constexpr in C++20
// template <InputIterator InIter,
//           OutputIterator<auto, const InIter::value_type&> OutIter,
//           Callable<auto, const InIter::value_type&, const InIter::value_type&> BinaryOperation>
//   requires Constructible<InIter::value_type, InIter::reference>
//         && OutputIterator<OutIter, BinaryOperation::result_type>
//         && MoveAssignable<InIter::value_type>
//         && CopyConstructible<BinaryOperation>
//   OutIter
//   adjacent_difference(InIter first, InIter last, OutIter result, BinaryOperation binary_op);

#include <numeric>
#include <functional>
#include <string>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

#if TEST_STD_VER > 17
struct rvalue_subtractable
{
    bool correctOperatorUsed = false;

    // make sure the predicate is passed an rvalue and an lvalue (so check that the first argument was moved)
    constexpr rvalue_subtractable operator()(rvalue_subtractable const&, rvalue_subtractable&& r) {
        r.correctOperatorUsed = true;
        return std::move(r);
    }
};

constexpr rvalue_subtractable operator-(rvalue_subtractable const&, rvalue_subtractable& rhs)
{
    rhs.correctOperatorUsed = false;
    return rhs;
}

constexpr rvalue_subtractable operator-(rvalue_subtractable const&, rvalue_subtractable&& rhs)
{
    rhs.correctOperatorUsed = true;
    return std::move(rhs);
}

constexpr void
test_use_move()
{
    const std::size_t size = 100;
    rvalue_subtractable arr[size];
    rvalue_subtractable res1[size];
    rvalue_subtractable res2[size];
    std::adjacent_difference(arr, arr + size, res1);
    std::adjacent_difference(arr, arr + size, res2, /*predicate=*/rvalue_subtractable());
    // start at 1 because the first element is not moved
    for (unsigned i = 1; i < size; ++i) assert(res1[i].correctOperatorUsed);
    for (unsigned i = 1; i < size; ++i) assert(res2[i].correctOperatorUsed);
}
#endif // TEST_STD_VER > 17

TEST_CONSTEXPR_CXX20 void test_string() {
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
    int ia[] = {15, 10, 6, 3, 1};
    int ir[] = {15, 25, 16, 9, 4};
    const unsigned s = sizeof(ia) / sizeof(ia[0]);
    int ib[s] = {0};
    OutIter r = std::adjacent_difference(InIter(ia), InIter(ia+s), OutIter(ib),
                                         std::plus<int>());
    assert(base(r) == ib + s);
    for (unsigned i = 0; i < s; ++i)
        assert(ib[i] == ir[i]);
}

#if TEST_STD_VER >= 11

class Y;

class X
{
    int i_;

    TEST_CONSTEXPR_CXX20 X& operator=(const X&);
public:
    TEST_CONSTEXPR_CXX20 explicit X(int i) : i_(i) {}
    TEST_CONSTEXPR_CXX20 X(const X& x) : i_(x.i_) {}
    TEST_CONSTEXPR_CXX20 X& operator=(X&& x)
    {
        i_ = x.i_;
        x.i_ = -1;
        return *this;
    }

    TEST_CONSTEXPR_CXX20 friend X operator-(const X& x, const X& y) {return X(x.i_ - y.i_);}

    friend class Y;
};

class Y
{
    int i_;

    TEST_CONSTEXPR_CXX20 Y& operator=(const Y&);
public:
    TEST_CONSTEXPR_CXX20 explicit Y(int i) : i_(i) {}
    TEST_CONSTEXPR_CXX20 Y(const Y& y) : i_(y.i_) {}
    TEST_CONSTEXPR_CXX20 void operator=(const X& x) {i_ = x.i_;}
};

#endif


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

#if TEST_STD_VER >= 11
    X x[3] = {X(1), X(2), X(3)};
    Y y[3] = {Y(1), Y(2), Y(3)};
    std::adjacent_difference(x, x+3, y, std::minus<X>());
#endif

#if TEST_STD_VER > 17
    test_use_move();
#endif // TEST_STD_VER > 17

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
