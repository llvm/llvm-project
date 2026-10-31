//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// move_iterator

// constexpr move_iterator();                   // constexpr since C++17
//   requires default_initializable<Iterator>;  // since C++20

#include <iterator>

#include <type_traits>
#include "test_macros.h"
#include "test_iterators.h"

#if TEST_STD_VER >= 20
class constable_iter {
public:
  using iterator_category = std::input_iterator_tag;
  using difference_type   = int;
  using value_type        = char;

  constable_iter() = default;

  template <class = void>
  const constable_iter& operator=(const constable_iter&) const;

  char operator*() const;

  const constable_iter& operator++() const;
  void operator++(int) const;

private:
  char payload_; // making `const constable_iter` value-initializable but not default-initializable
};
static_assert(std::is_default_constructible_v<const constable_iter>);
static_assert(!std::default_initializable<const constable_iter>);
static_assert(std::input_iterator<const constable_iter>);

static_assert(std::is_default_constructible_v<std::move_iterator<forward_iterator<int*>>>);
static_assert(!std::is_default_constructible_v<std::move_iterator<cpp20_input_iterator<int*>>>);
static_assert(!std::is_default_constructible_v<std::move_iterator<const constable_iter>>);
#endif

template <class It>
void test() {
    std::move_iterator<It> r;
    (void)r;
}

int main(int, char**) {
  // we don't have a test iterator that is both input and default-constructible, so not testing that case
  test<forward_iterator<char*> >();
  test<bidirectional_iterator<char*> >();
  test<random_access_iterator<char*> >();
  test<char*>();

#if TEST_STD_VER > 14
  {
    constexpr std::move_iterator<const char *> it;
    (void)it;
  }
#endif

  return 0;
}
