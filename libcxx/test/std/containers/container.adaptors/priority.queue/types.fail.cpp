//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <queue>

// template <class T, class Container = vector<T>,
//           class Compare = less<typename Container::value_type>>
// class priority_queue
// {
// public:
//     typedef Container                                container_type;
//     typedef typename container_type::value_type      value_type;
//     typedef typename container_type::reference       reference;
//     typedef typename container_type::const_reference const_reference;
//     typedef typename container_type::size_type       size_type;
//
// protected:
//     container_type c;
//     Compare comp;

#include <queue>
#include <cassert>
#include <type_traits>

int main()
{
//  LWG#2566 says that the first template param must match the second one's value type
    std::priority_queue<double, std::deque<int>> t;
}
