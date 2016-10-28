//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <vector>

// template <class InputIter> vector(InputIter first, InputIter last);

#include <vector>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "test_allocator.h"
#include "min_allocator.h"
#include "asan_testing.h"

template <class C, class Iterator>
void
test(Iterator first, Iterator last)
{
    C c(first, last);
    LIBCPP_ASSERT(c.__invariants());
    assert(c.size() == std::distance(first, last));
    LIBCPP_ASSERT(is_contiguous_container_asan_correct(c));
    for (typename C::const_iterator i = c.cbegin(), e = c.cend(); i != e; ++i, ++first)
        assert(*i == *first);
}

int main()
{
    int a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 1, 0};
    int* an = a + sizeof(a)/sizeof(a[0]);
    test<std::vector<int> >(input_iterator<const int*>(a), input_iterator<const int*>(an));
    test<std::vector<int> >(forward_iterator<const int*>(a), forward_iterator<const int*>(an));
    test<std::vector<int> >(bidirectional_iterator<const int*>(a), bidirectional_iterator<const int*>(an));
    test<std::vector<int> >(random_access_iterator<const int*>(a), random_access_iterator<const int*>(an));
    test<std::vector<int> >(a, an);

    test<std::vector<int, limited_allocator<int, 63> > >(input_iterator<const int*>(a), input_iterator<const int*>(an));
    // Add 1 for implementations that dynamically allocate a container proxy.
    test<std::vector<int, limited_allocator<int, 18 + 1> > >(forward_iterator<const int*>(a), forward_iterator<const int*>(an));
    test<std::vector<int, limited_allocator<int, 18 + 1> > >(bidirectional_iterator<const int*>(a), bidirectional_iterator<const int*>(an));
    test<std::vector<int, limited_allocator<int, 18 + 1> > >(random_access_iterator<const int*>(a), random_access_iterator<const int*>(an));
    test<std::vector<int, limited_allocator<int, 18 + 1> > >(a, an);
#if TEST_STD_VER >= 11
    test<std::vector<int, min_allocator<int>> >(input_iterator<const int*>(a), input_iterator<const int*>(an));
    test<std::vector<int, min_allocator<int>> >(forward_iterator<const int*>(a), forward_iterator<const int*>(an));
    test<std::vector<int, min_allocator<int>> >(bidirectional_iterator<const int*>(a), bidirectional_iterator<const int*>(an));
    test<std::vector<int, min_allocator<int>> >(random_access_iterator<const int*>(a), random_access_iterator<const int*>(an));
    test<std::vector<int> >(a, an);
#endif
}
