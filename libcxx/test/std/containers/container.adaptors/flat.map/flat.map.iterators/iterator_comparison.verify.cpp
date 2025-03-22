//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <flat_map>

// For underlying iterators i and j, i <=> j must be well-formed and return std::strong_ordering.

#include <flat_map>
#include <functional>
#include <iterator>
#include <type_traits>

#include "MinSequenceContainer.h"
#include "test_iterators.h"

template <class It>
class bad_3way_random_access_iterator {
  template <class U>
  friend class bad_3way_random_access_iterator;

public:
  using iterator_category = std::random_access_iterator_tag;
  using value_type        = typename std::iterator_traits<It>::value_type;
  using difference_type   = typename std::iterator_traits<It>::difference_type;
  using pointer           = It;
  using reference         = typename std::iterator_traits<It>::reference;

  bad_3way_random_access_iterator();
  explicit bad_3way_random_access_iterator(It);

  template <class U>
  bad_3way_random_access_iterator(const bad_3way_random_access_iterator<U>&);

  template <class U, class = typename std::enable_if<std::is_default_constructible<U>::value>::type>
  bad_3way_random_access_iterator(bad_3way_random_access_iterator<U>&&);

  reference operator*() const;
  reference operator[](difference_type) const;

  bad_3way_random_access_iterator& operator++();
  bad_3way_random_access_iterator& operator--();
  bad_3way_random_access_iterator operator++(int);
  bad_3way_random_access_iterator operator--(int);

  bad_3way_random_access_iterator& operator+=(difference_type);
  bad_3way_random_access_iterator& operator-=(difference_type);
  friend bad_3way_random_access_iterator operator+(bad_3way_random_access_iterator, difference_type);
  friend bad_3way_random_access_iterator operator+(difference_type, bad_3way_random_access_iterator);
  friend bad_3way_random_access_iterator operator-(bad_3way_random_access_iterator, difference_type);
  friend difference_type operator-(bad_3way_random_access_iterator, bad_3way_random_access_iterator);

  friend bool operator==(const bad_3way_random_access_iterator&, const bad_3way_random_access_iterator&);
  friend std::weak_ordering operator<=>(const bad_3way_random_access_iterator&, const bad_3way_random_access_iterator&);
};

void test() {
  {
    using KeyCont = MinSequenceContainer<int, random_access_iterator<int*>, random_access_iterator<const int*>>;
    using FMap    = std::flat_map<int, int, std::less<int>, KeyCont, MinSequenceContainer<int>>;
    FMap m;
    // expected-error@*:* {{static assertion failed: random accesss iterator not supporting three-way comparison is invalid for container}}
    // expected-error@*:* {{invalid operands to binary expression}}
    (void)(m.begin() <=> m.begin());
  }
  {
    using KeyCont =
        MinSequenceContainer<int, bad_3way_random_access_iterator<int*>, bad_3way_random_access_iterator<const int*>>;
    using FMap = std::flat_map<int, int, std::less<int>, KeyCont, MinSequenceContainer<int>>;
    FMap m;
    // expected-error-re@*:* {{static assertion failed due to requirement 'is_same_v<std::weak_ordering, std::strong_ordering>'{{.*}}three-way comparison between random accesss container iterators must return std::strong_ordering}}
    // expected-error-re@*:* {{no viable conversion from returned value of type{{.*}}}}
    (void)(m.begin() <=> m.begin());
  }
}
