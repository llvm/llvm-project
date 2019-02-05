//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

// <forward_list>

// template <class T, class Allocator, class U>
//   void erase(forward_list<T, Allocator>& c, const U& value);
  

#include <forward_list>
#include <optional>

#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"

template <class S, class U>
void
test0(S s,  U val, S expected)
{
    ASSERT_SAME_TYPE(void, decltype(std::erase(s, val)));
    std::erase(s, val);
    assert(s == expected);
}

template <class S>
void test()
{

    test0(S(), 1, S());

    test0(S({1}), 1, S());
    test0(S({1}), 2, S({1}));

    test0(S({1,2}), 1, S({2}));
    test0(S({1,2}), 2, S({1}));
    test0(S({1,2}), 3, S({1,2}));
    test0(S({1,1}), 1, S());
    test0(S({1,1}), 3, S({1,1}));

    test0(S({1,2,3}), 1, S({2,3}));
    test0(S({1,2,3}), 2, S({1,3}));
    test0(S({1,2,3}), 3, S({1,2}));
    test0(S({1,2,3}), 4, S({1,2,3}));

    test0(S({1,1,1}), 1, S());
    test0(S({1,1,1}), 2, S({1,1,1}));
    test0(S({1,1,2}), 1, S({2}));
    test0(S({1,1,2}), 2, S({1,1}));
    test0(S({1,1,2}), 3, S({1,1,2}));
    test0(S({1,2,2}), 1, S({2,2}));
    test0(S({1,2,2}), 2, S({1}));
    test0(S({1,2,2}), 3, S({1,2,2}));

//  Test cross-type erasure
    using opt = std::optional<typename S::value_type>;
    test0(S({1,2,1}), opt(),  S({1,2,1}));
    test0(S({1,2,1}), opt(1), S({2}));
    test0(S({1,2,1}), opt(2), S({1,1}));
    test0(S({1,2,1}), opt(3), S({1,2,1}));
}

int main(int, char**)
{
    test<std::forward_list<int>>();
    test<std::forward_list<int, min_allocator<int>>> ();
    test<std::forward_list<int, test_allocator<int>>> ();

    test<std::forward_list<long>>();
    test<std::forward_list<double>>();

  return 0;
}
