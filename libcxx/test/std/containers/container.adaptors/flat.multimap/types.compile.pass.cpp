//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

//  using key_type               = Key;
//  using mapped_type            = T;
//  using value_type             = pair<key_type, mapped_type>;
//  using key_compare            = Compare;
//  using reference              = pair<const key_type&, mapped_type&>;
//  using const_reference        = pair<const key_type&, const mapped_type&>;
//  using size_type              = size_t;
//  using difference_type        = ptrdiff_t;
//  using iterator               = implementation-defined; // see [container.requirements]
//  using const_iterator         = implementation-defined; // see [container.requirements]
//  using reverse_iterator       = std::reverse_iterator<iterator>;
//  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
//  using key_container_type     = KeyContainer;
//  using mapped_container_type  = MappedContainer;

//  class value_compare;

//  struct containers {
//    key_container_type keys;
//    mapped_container_type values;
//  };

#include <concepts>
#include <deque>
#include <flat_map>
#include <functional>
#include <ranges>
#include <string>
#include <vector>
#include "min_allocator.h"

void test() {
  {
    using M = std::flat_multimap<int, double>;
    static_assert(std::is_same_v<typename M::key_type, int>);
    static_assert(std::is_same_v<typename M::mapped_type, double>);
    static_assert(std::is_same_v<typename M::value_type, std::pair<int, double>>);
    static_assert(std::is_same_v<typename M::key_compare, std::less<int>>);
    static_assert(std::is_same_v<typename M::reference, std::pair<const int&, double&>>);
    static_assert(std::is_same_v<typename M::const_reference, std::pair<const int&, const double&>>);
    static_assert(std::is_same_v<typename M::size_type, size_t>);
    static_assert(std::is_same_v<typename M::difference_type, ptrdiff_t>);
    static_assert(requires { typename M::iterator; });
    static_assert(requires { typename M::const_iterator; });
    static_assert(std::is_same_v<typename M::reverse_iterator, std::reverse_iterator<typename M::iterator>>);
    static_assert(
        std::is_same_v<typename M::const_reverse_iterator, std::reverse_iterator<typename M::const_iterator>>);
    static_assert(std::is_same_v<typename M::key_container_type, std::vector<int>>);
    static_assert(std::is_same_v<typename M::mapped_container_type, std::vector<double>>);
    static_assert(requires { typename M::value_compare; });
    static_assert(requires { typename M::containers; });
    static_assert(std::is_same_v<decltype(M::containers::keys), std::vector<int>>);
    static_assert(std::is_same_v<decltype(M::containers::values), std::vector<double>>);
  }

  {
    struct A {};
    struct Compare {
      bool operator()(const std::string&, const std::string&) const;
    };
    using M = std::flat_multimap<std::string, A, Compare, std::deque<std::string>, std::deque<A>>;
    static_assert(std::is_same_v<typename M::key_type, std::string>);
    static_assert(std::is_same_v<typename M::mapped_type, A>);
    static_assert(std::is_same_v<typename M::value_type, std::pair<std::string, A>>);
    static_assert(std::is_same_v<typename M::key_compare, Compare>);
    static_assert(std::is_same_v<typename M::reference, std::pair<const std::string&, A&>>);
    static_assert(std::is_same_v<typename M::const_reference, std::pair<const std::string&, const A&>>);
    static_assert(std::is_same_v<typename M::size_type, size_t>);
    static_assert(std::is_same_v<typename M::difference_type, ptrdiff_t>);
    static_assert(requires { typename M::iterator; });
    static_assert(requires { typename M::const_iterator; });
    static_assert(std::is_same_v<typename M::reverse_iterator, std::reverse_iterator<typename M::iterator>>);
    static_assert(
        std::is_same_v<typename M::const_reverse_iterator, std::reverse_iterator<typename M::const_iterator>>);
    static_assert(std::is_same_v<typename M::key_container_type, std::deque<std::string>>);
    static_assert(std::is_same_v<typename M::mapped_container_type, std::deque<A>>);
    static_assert(requires { typename M::value_compare; });
    static_assert(requires { typename M::containers; });
    static_assert(std::is_same_v<decltype(M::containers::keys), std::deque<std::string>>);
    static_assert(std::is_same_v<decltype(M::containers::values), std::deque<A>>);
  }
  {
    using C = std::flat_multimap<int, short>;
    static_assert(std::is_same_v<C::key_type, int>);
    static_assert(std::is_same_v<C::mapped_type, short>);
    static_assert(std::is_same_v<C::value_type, std::pair<int, short>>);
    static_assert(std::is_same_v<C::key_compare, std::less<int>>);
    static_assert(!std::is_same_v<C::value_compare, std::less<int>>);
    static_assert(std::is_same_v<C::reference, std::pair<const int&, short&>>);
    static_assert(std::is_same_v<C::const_reference, std::pair<const int&, const short&>>);
    static_assert(std::random_access_iterator<C::iterator>);
    static_assert(std::random_access_iterator<C::const_iterator>);
    static_assert(std::random_access_iterator<C::reverse_iterator>);
    static_assert(std::random_access_iterator<C::const_reverse_iterator>);
    static_assert(std::is_same_v<C::reverse_iterator, std::reverse_iterator<C::iterator>>);
    static_assert(std::is_same_v<C::const_reverse_iterator, std::reverse_iterator<C::const_iterator>>);
    static_assert(std::is_same_v<C::size_type, std::size_t>);
    static_assert(std::is_same_v<C::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<C::key_container_type, std::vector<int>>);
    static_assert(std::is_same_v<C::mapped_container_type, std::vector<short>>);
  }
  {
    using C = std::flat_multimap<short, int, std::greater<long>, std::deque<short, min_allocator<short>>>;
    static_assert(std::is_same_v<C::key_type, short>);
    static_assert(std::is_same_v<C::mapped_type, int>);
    static_assert(std::is_same_v<C::value_type, std::pair<short, int>>);
    static_assert(std::is_same_v<C::key_compare, std::greater<long>>);
    static_assert(!std::is_same_v<C::value_compare, std::greater<long>>);
    static_assert(std::is_same_v<C::reference, std::pair<const short&, int&>>);
    static_assert(std::is_same_v<C::const_reference, std::pair<const short&, const int&>>);
    static_assert(std::random_access_iterator<C::iterator>);
    static_assert(std::random_access_iterator<C::const_iterator>);
    static_assert(std::random_access_iterator<C::reverse_iterator>);
    static_assert(std::random_access_iterator<C::const_reverse_iterator>);
    static_assert(std::is_same_v<C::reverse_iterator, std::reverse_iterator<C::iterator>>);
    static_assert(std::is_same_v<C::const_reverse_iterator, std::reverse_iterator<C::const_iterator>>);
    // size_type is invariably size_t
    static_assert(std::is_same_v<C::size_type, std::size_t>);
    static_assert(std::is_same_v<C::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<C::key_container_type, std::deque<short, min_allocator<short>>>);
    static_assert(std::is_same_v<C::mapped_container_type, std::vector<int>>);
  }
}
