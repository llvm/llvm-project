//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// using key_type                  = Key;
// using value_type                = Key;
// using key_compare               = Compare;
// using value_compare             = Compare;
// using reference                 = value_type&;
// using const_reference           = const value_type&;
// using size_type                 = typename KeyContainer::size_type;
// using difference_type           = typename KeyContainer::difference_type;
// using iterator                  = implementation-defined;  // see [container.requirements]
// using const_iterator            = implementation-defined;  // see [container.requirements]
// using reverse_iterator          = std::reverse_iterator<iterator>;
// using const_reverse_iterator    = std::reverse_iterator<const_iterator>;
// using container_type            = KeyContainer;

#include <concepts>
#include <deque>
#include <flat_set>
#include <functional>
#include <ranges>
#include <string>
#include <vector>
#include "min_allocator.h"

void test() {
  {
    using M = std::flat_set<int>;
    static_assert(std::is_same_v<typename M::key_type, int>);
    static_assert(std::is_same_v<typename M::value_type, int>);
    static_assert(std::is_same_v<typename M::key_compare, std::less<int>>);
    static_assert(std::is_same_v<typename M::value_compare, std::less<int>>);
    static_assert(std::is_same_v<typename M::reference, int&>);
    static_assert(std::is_same_v<typename M::const_reference, const int&>);
    static_assert(std::is_same_v<typename M::size_type, size_t>);
    static_assert(std::is_same_v<typename M::difference_type, ptrdiff_t>);
    static_assert(requires { typename M::iterator; });
    static_assert(requires { typename M::const_iterator; });
    static_assert(std::is_same_v<typename M::reverse_iterator, std::reverse_iterator<typename M::iterator>>);
    static_assert(
        std::is_same_v<typename M::const_reverse_iterator, std::reverse_iterator<typename M::const_iterator>>);
    static_assert(std::is_same_v<typename M::container_type, std::vector<int>>);
    static_assert(requires { typename M::value_compare; });
  }

  {
    struct A {};
    struct Compare {
      bool operator()(const std::string&, const std::string&) const;
    };
    using M = std::flat_set<std::string, Compare, std::deque<std::string>>;
    static_assert(std::is_same_v<typename M::key_type, std::string>);
    static_assert(std::is_same_v<typename M::value_type, std::string>);
    static_assert(std::is_same_v<typename M::key_compare, Compare>);
    static_assert(std::is_same_v<typename M::value_compare, Compare>);
    static_assert(std::is_same_v<typename M::reference, std::string&>);
    static_assert(std::is_same_v<typename M::const_reference, const std::string&>);
    static_assert(std::is_same_v<typename M::size_type, size_t>);
    static_assert(std::is_same_v<typename M::difference_type, ptrdiff_t>);
    static_assert(requires { typename M::iterator; });
    static_assert(requires { typename M::const_iterator; });
    static_assert(std::is_same_v<typename M::reverse_iterator, std::reverse_iterator<typename M::iterator>>);
    static_assert(
        std::is_same_v<typename M::const_reverse_iterator, std::reverse_iterator<typename M::const_iterator>>);
    static_assert(std::is_same_v<typename M::container_type, std::deque<std::string>>);
  }
  {
    using C = std::flat_set<short, std::greater<long>, std::deque<short, min_allocator<short>>>;
    static_assert(std::is_same_v<C::key_type, short>);
    static_assert(std::is_same_v<C::value_type, short>);
    static_assert(std::is_same_v<C::key_compare, std::greater<long>>);
    static_assert(std::is_same_v<C::value_compare, std::greater<long>>);
    static_assert(std::is_same_v<C::reference, short&>);
    static_assert(std::is_same_v<C::const_reference, const short&>);
    static_assert(std::random_access_iterator<C::iterator>);
    static_assert(std::random_access_iterator<C::const_iterator>);
    static_assert(std::random_access_iterator<C::reverse_iterator>);
    static_assert(std::random_access_iterator<C::const_reverse_iterator>);
    static_assert(std::is_same_v<C::reverse_iterator, std::reverse_iterator<C::iterator>>);
    static_assert(std::is_same_v<C::const_reverse_iterator, std::reverse_iterator<C::const_iterator>>);
    // size_type is invariably size_t
    static_assert(std::is_same_v<C::size_type, std::size_t>);
    static_assert(std::is_same_v<C::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<C::container_type, std::deque<short, min_allocator<short>>>);
  }
}
