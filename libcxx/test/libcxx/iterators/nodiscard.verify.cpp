//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: libcpp-hardening-mode=none

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ABI_BOUNDED_ITERATORS
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ABI_BOUNDED_ITERATORS_IN_STD_ARRAY

// check that <iterator> functions are marked [[nodiscard]]

#include <array>
#include <initializer_list>
#include <iterator>
#include <sstream>
#include <string_view>
#include <vector>

#include "test_macros.h"
#include "test_iterators.h"

void test() {
  int cArr[] = {94, 82, 49};
  std::vector<int> cont;
  const std::vector<int> cCont;
#if TEST_STD_VER >= 11
  std::initializer_list<int> il;
#endif
#if !defined(TEST_HAS_NO_LOCALIZATION)
  std::stringstream ss;
#endif

  { // Access
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::begin(cArr);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::end(cArr);

#if TEST_STD_VER >= 11
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::begin(cont);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::begin(cCont);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::end(cont);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::end(cCont);
#endif
#if TEST_STD_VER >= 14
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::cbegin(cCont);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::cend(cCont);
#endif
  }

  {
    std::back_insert_iterator<std::vector<int> > it(cont);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    *it;
  }

  { // __bounded_iter
    typedef std::basic_string_view<char> Container;
    ASSERT_SAME_TYPE(Container::const_iterator, std::__bounded_iter<const char*>);

    Container::const_iterator it;

    std::pointer_traits<Container::const_iterator> pt;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    pt.to_address(it);
  }

#if TEST_STD_VER >= 20
  {
    std::common_iterator<int*, sentinel_wrapper<int*>> it;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    *it;
  }
#endif

#if TEST_STD_VER >= 20
  {
    std::counted_iterator it{random_access_iterator<int*>{cArr}, 3};

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it.base();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::move(it).base();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it.count();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    *it;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    *std::as_const(it);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it + 2;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    2 + it;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it - 2;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it - it;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it - std::default_sentinel;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::default_sentinel - it;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it[2];
  }
#endif

#if TEST_STD_VER >= 17
  {
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::data(cont);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::data(cCont);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::data(cArr);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::data(il);
  }
#endif

  {
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::distance(cont.begin(), cont.end());
#if TEST_STD_VER >= 20
    cpp17_input_iterator<int*> it{cArr};
    sentinel_wrapper<cpp17_input_iterator<int*>> st;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::distance(it, st);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::distance(cont.begin(), cont.end());
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::distance(std::move(cont));
#endif
  }

#if TEST_STD_VER >= 17
  { // Empty
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::empty(cont);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::empty(cArr);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::empty(il);
  }
#endif

  {
    std::front_insert_iterator<std::vector<int> > it(cont);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    *it;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::front_inserter(cont);
  }

  {
    std::insert_iterator<std::vector<int> > it(cont, cont.begin());

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    *it;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::inserter(cont, cont.begin());
  }

  {
    std::istream_iterator<char> it;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    *it;
  }

#if !defined(TEST_HAS_NO_LOCALIZATION)
  {
    std::istreambuf_iterator<char> it(ss);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    *it;
  }
#endif

#if TEST_STD_VER >= 11
  {
    std::move_iterator<random_access_iterator<int*>> it;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it.base();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::move(it).base();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    *it;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it[0];

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it + 1;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it - 1;

#  if TEST_STD_VER >= 20
    std::move_sentinel<random_access_iterator<int*>> st;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    st - it;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it - st;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    iter_move(it);
#  endif

    std::move_iterator<random_access_iterator<int*>> otherIt;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it - otherIt;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    otherIt - it;

#  if TEST_STD_VER >= 20
    std::iter_difference_t<random_access_iterator<int*>> diff = st - it;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    diff + it;
#  endif

    int* i = nullptr;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::make_move_iterator(i);
  }
#endif

#if TEST_STD_VER >= 20
  {
    std::move_sentinel<int*> st;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    st.base();
  }
#endif

  {
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::next(cArr);

#if TEST_STD_VER >= 20
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::next(cArr);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::next(cont.begin(), 2);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::next(cont.begin(), cont.end());
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::next(cont.begin(), 2, cont.end());
#endif
  }

#if !defined(TEST_HAS_NO_LOCALIZATION)
  {
    std::ostream_iterator<char> it(ss);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    *it;
  }
#endif

#if !defined(TEST_HAS_NO_LOCALIZATION)
  {
    std::ostreambuf_iterator<char> it(ss);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    *it;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it.failed();
  }
#endif

  {
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::prev(cArr, 2);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::prev(cArr);

#if TEST_STD_VER >= 20
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::prev(cArr);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::prev(cont.end(), 2);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::prev(cont.end(), 2, cont.begin());
#endif
  }

#if TEST_STD_VER >= 14
  {
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::rbegin(cArr);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::rend(cArr);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::rbegin(il);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::rend(il);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::rbegin(cont);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::rbegin(cCont);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::rend(cont);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::rend(cCont);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::crbegin(cCont);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::crend(cCont);
  }
#endif

  {
    std::reverse_iterator<random_access_iterator<const char*> > it;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it.base();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    *it;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it - 1;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it[0];

#if _LIBCPP_STD_VER >= 20
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    iter_move(it);
#endif

    std::reverse_iterator<random_access_iterator<const char*> > otherIt;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    otherIt - it;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it - otherIt;

#if TEST_STD_VER >= 14
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::make_reverse_iterator(cont.end());
#endif
  }

#if TEST_STD_VER >= 17
  {
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::size(cArr);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::size(cont);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::size(cCont);
  }
#endif

  { // __static_bounded_iter
    typedef std::array<int, 94> Container;
    ASSERT_SAME_TYPE(Container::iterator, std::__static_bounded_iter<int*, 94>);

    Container::iterator it;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    *it;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it[0];

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it + 1;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    1 + it;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it - 1;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it - it;

    std::pointer_traits<Container::iterator> pt;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    pt.to_address(it);
  }

  { // __wrap_iter
    typedef std::vector<int> Container;
    ASSERT_SAME_TYPE(Container::iterator, std::__wrap_iter<int*>);

    Container::iterator it;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    *it;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it + 1;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it[0];

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it.base();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it - it;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    1 + it;

    std::pointer_traits<Container::iterator> pt;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    pt.to_address(it);
  }
}
