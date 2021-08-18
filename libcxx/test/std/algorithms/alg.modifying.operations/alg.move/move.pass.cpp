//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator InIter, typename OutIter>
//   requires OutputIterator<OutIter, RvalueOf<InIter::reference>::type>
//   OutIter
//   move(InIter first, InIter last, OutIter result);

// Older compilers don't support std::is_constant_evaluated
// UNSUPPORTED: clang-4, clang-5, clang-6, clang-7, clang-8
// UNSUPPORTED: apple-clang-9, apple-clang-10, apple-clang-11

#include <algorithm>
#include <cassert>
#include <memory>

#include "test_macros.h"
#include "test_iterators.h"

template <class InIter, class OutIter>
TEST_CONSTEXPR_CXX17 bool
test()
{
    const unsigned N = 1000;
    int ia[N] = {};
    for (unsigned i = 0; i < N; ++i)
        ia[i] = i;
    int ib[N] = {0};

    OutIter r = std::move(InIter(ia), InIter(ia+N), OutIter(ib));
    assert(base(r) == ib+N);
    for (unsigned i = 0; i < N; ++i)
        assert(ia[i] == ib[i]);

    return true;
}

#if TEST_STD_VER >= 11
template <class InIter, class OutIter>
void
test1()
{
    const unsigned N = 100;
    std::unique_ptr<int> ia[N];
    for (unsigned i = 0; i < N; ++i)
        ia[i].reset(new int(i));
    std::unique_ptr<int> ib[N];

    OutIter r = std::move(InIter(ia), InIter(ia+N), OutIter(ib));
    assert(base(r) == ib+N);
    for (unsigned i = 0; i < N; ++i)
        assert(*ib[i] == static_cast<int>(i));
}
#endif

int main(int, char**)
{
    test<cpp17_input_iterator<const int*>, output_iterator<int*> >();
    test<cpp17_input_iterator<const int*>, forward_iterator<int*> >();
    test<cpp17_input_iterator<const int*>, bidirectional_iterator<int*> >();
    test<cpp17_input_iterator<const int*>, random_access_iterator<int*> >();
    test<cpp17_input_iterator<const int*>, int*>();

    test<forward_iterator<const int*>, output_iterator<int*> >();
    test<forward_iterator<const int*>, forward_iterator<int*> >();
    test<forward_iterator<const int*>, bidirectional_iterator<int*> >();
    test<forward_iterator<const int*>, random_access_iterator<int*> >();
    test<forward_iterator<const int*>, int*>();

    test<bidirectional_iterator<const int*>, output_iterator<int*> >();
    test<bidirectional_iterator<const int*>, forward_iterator<int*> >();
    test<bidirectional_iterator<const int*>, bidirectional_iterator<int*> >();
    test<bidirectional_iterator<const int*>, random_access_iterator<int*> >();
    test<bidirectional_iterator<const int*>, int*>();

    test<random_access_iterator<const int*>, output_iterator<int*> >();
    test<random_access_iterator<const int*>, forward_iterator<int*> >();
    test<random_access_iterator<const int*>, bidirectional_iterator<int*> >();
    test<random_access_iterator<const int*>, random_access_iterator<int*> >();
    test<random_access_iterator<const int*>, int*>();

    test<const int*, output_iterator<int*> >();
    test<const int*, forward_iterator<int*> >();
    test<const int*, bidirectional_iterator<int*> >();
    test<const int*, random_access_iterator<int*> >();
    test<const int*, int*>();

#if TEST_STD_VER >= 11
    test1<cpp17_input_iterator<std::unique_ptr<int>*>, output_iterator<std::unique_ptr<int>*> >();
    test1<cpp17_input_iterator<std::unique_ptr<int>*>, forward_iterator<std::unique_ptr<int>*> >();
    test1<cpp17_input_iterator<std::unique_ptr<int>*>, bidirectional_iterator<std::unique_ptr<int>*> >();
    test1<cpp17_input_iterator<std::unique_ptr<int>*>, random_access_iterator<std::unique_ptr<int>*> >();
    test1<cpp17_input_iterator<std::unique_ptr<int>*>, std::unique_ptr<int>*>();

    test1<forward_iterator<std::unique_ptr<int>*>, output_iterator<std::unique_ptr<int>*> >();
    test1<forward_iterator<std::unique_ptr<int>*>, forward_iterator<std::unique_ptr<int>*> >();
    test1<forward_iterator<std::unique_ptr<int>*>, bidirectional_iterator<std::unique_ptr<int>*> >();
    test1<forward_iterator<std::unique_ptr<int>*>, random_access_iterator<std::unique_ptr<int>*> >();
    test1<forward_iterator<std::unique_ptr<int>*>, std::unique_ptr<int>*>();

    test1<bidirectional_iterator<std::unique_ptr<int>*>, output_iterator<std::unique_ptr<int>*> >();
    test1<bidirectional_iterator<std::unique_ptr<int>*>, forward_iterator<std::unique_ptr<int>*> >();
    test1<bidirectional_iterator<std::unique_ptr<int>*>, bidirectional_iterator<std::unique_ptr<int>*> >();
    test1<bidirectional_iterator<std::unique_ptr<int>*>, random_access_iterator<std::unique_ptr<int>*> >();
    test1<bidirectional_iterator<std::unique_ptr<int>*>, std::unique_ptr<int>*>();

    test1<random_access_iterator<std::unique_ptr<int>*>, output_iterator<std::unique_ptr<int>*> >();
    test1<random_access_iterator<std::unique_ptr<int>*>, forward_iterator<std::unique_ptr<int>*> >();
    test1<random_access_iterator<std::unique_ptr<int>*>, bidirectional_iterator<std::unique_ptr<int>*> >();
    test1<random_access_iterator<std::unique_ptr<int>*>, random_access_iterator<std::unique_ptr<int>*> >();
    test1<random_access_iterator<std::unique_ptr<int>*>, std::unique_ptr<int>*>();

    test1<std::unique_ptr<int>*, output_iterator<std::unique_ptr<int>*> >();
    test1<std::unique_ptr<int>*, forward_iterator<std::unique_ptr<int>*> >();
    test1<std::unique_ptr<int>*, bidirectional_iterator<std::unique_ptr<int>*> >();
    test1<std::unique_ptr<int>*, random_access_iterator<std::unique_ptr<int>*> >();
    test1<std::unique_ptr<int>*, std::unique_ptr<int>*>();
#endif // TEST_STD_VER >= 11

#if TEST_STD_VER > 17
    test<cpp17_input_iterator<const int*>, contiguous_iterator<int*>>();
    test<forward_iterator<const int*>, contiguous_iterator<int*>>();
    test<bidirectional_iterator<const int*>, contiguous_iterator<int*>>();
    test<random_access_iterator<const int*>, contiguous_iterator<int*>>();
    test<const int*, contiguous_iterator<int*>>();
    test<contiguous_iterator<const int*>, output_iterator<int*>>();
    test<contiguous_iterator<const int*>, forward_iterator<int*>>();
    test<contiguous_iterator<const int*>, bidirectional_iterator<int*>>();
    test<contiguous_iterator<const int*>, random_access_iterator<int*>>();
    test<contiguous_iterator<const int*>, int*>();
    test<contiguous_iterator<const int*>, contiguous_iterator<int*>>();

    test1<cpp17_input_iterator<std::unique_ptr<int>*>, contiguous_iterator<std::unique_ptr<int>*>>();
    test1<forward_iterator<std::unique_ptr<int>*>, contiguous_iterator<std::unique_ptr<int>*>>();
    test1<bidirectional_iterator<std::unique_ptr<int>*>, contiguous_iterator<std::unique_ptr<int>*>>();
    test1<random_access_iterator<std::unique_ptr<int>*>, contiguous_iterator<std::unique_ptr<int>*>>();
    test1<std::unique_ptr<int>*, contiguous_iterator<std::unique_ptr<int>*>>();
    test1<contiguous_iterator<std::unique_ptr<int>*>, output_iterator<std::unique_ptr<int>*>>();
    test1<contiguous_iterator<std::unique_ptr<int>*>, forward_iterator<std::unique_ptr<int>*>>();
    test1<contiguous_iterator<std::unique_ptr<int>*>, bidirectional_iterator<std::unique_ptr<int>*>>();
    test1<contiguous_iterator<std::unique_ptr<int>*>, random_access_iterator<std::unique_ptr<int>*>>();
    test1<contiguous_iterator<std::unique_ptr<int>*>, std::unique_ptr<int>*>();
    test1<contiguous_iterator<std::unique_ptr<int>*>, contiguous_iterator<std::unique_ptr<int>*>>();

    static_assert(test<cpp17_input_iterator<const int*>, cpp17_input_iterator<int*> >());
    static_assert(test<cpp17_input_iterator<const int*>, forward_iterator<int*> >());
    static_assert(test<cpp17_input_iterator<const int*>, bidirectional_iterator<int*> >());
    static_assert(test<cpp17_input_iterator<const int*>, random_access_iterator<int*> >());
    static_assert(test<cpp17_input_iterator<const int*>, contiguous_iterator<int*> >());
    static_assert(test<cpp17_input_iterator<const int*>, int*>());

    static_assert(test<forward_iterator<const int*>, cpp17_input_iterator<int*> >());
    static_assert(test<forward_iterator<const int*>, forward_iterator<int*> >());
    static_assert(test<forward_iterator<const int*>, bidirectional_iterator<int*> >());
    static_assert(test<forward_iterator<const int*>, random_access_iterator<int*> >());
    static_assert(test<forward_iterator<const int*>, contiguous_iterator<int*> >());
    static_assert(test<forward_iterator<const int*>, int*>());

    static_assert(test<bidirectional_iterator<const int*>, cpp17_input_iterator<int*> >());
    static_assert(test<bidirectional_iterator<const int*>, forward_iterator<int*> >());
    static_assert(test<bidirectional_iterator<const int*>, bidirectional_iterator<int*> >());
    static_assert(test<bidirectional_iterator<const int*>, random_access_iterator<int*> >());
    static_assert(test<bidirectional_iterator<const int*>, contiguous_iterator<int*> >());
    static_assert(test<bidirectional_iterator<const int*>, int*>());

    static_assert(test<random_access_iterator<const int*>, cpp17_input_iterator<int*> >());
    static_assert(test<random_access_iterator<const int*>, forward_iterator<int*> >());
    static_assert(test<random_access_iterator<const int*>, bidirectional_iterator<int*> >());
    static_assert(test<random_access_iterator<const int*>, random_access_iterator<int*> >());
    static_assert(test<random_access_iterator<const int*>, contiguous_iterator<int*> >());
    static_assert(test<random_access_iterator<const int*>, int*>());

    static_assert(test<contiguous_iterator<const int*>, cpp17_input_iterator<int*> >());
    static_assert(test<contiguous_iterator<const int*>, forward_iterator<int*> >());
    static_assert(test<contiguous_iterator<const int*>, bidirectional_iterator<int*> >());
    static_assert(test<contiguous_iterator<const int*>, random_access_iterator<int*> >());
    static_assert(test<contiguous_iterator<const int*>, contiguous_iterator<int*> >());
    static_assert(test<contiguous_iterator<const int*>, int*>());

    static_assert(test<const int*, cpp17_input_iterator<int*> >());
    static_assert(test<const int*, forward_iterator<int*> >());
    static_assert(test<const int*, bidirectional_iterator<int*> >());
    static_assert(test<const int*, random_access_iterator<int*> >());
    static_assert(test<const int*, contiguous_iterator<int*> >());
    static_assert(test<const int*, int*>());
#endif // TEST_STD_VER > 17

  return 0;
}
