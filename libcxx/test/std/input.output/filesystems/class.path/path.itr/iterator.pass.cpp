//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <filesystem>

// class path

// template <class Source>
//      path(const Source& source);
// template <class InputIterator>
//      path(InputIterator first, InputIterator last);


#include "filesystem_include.h"
#include <iterator>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "filesystem_test_helper.h"



template <class It>
std::reverse_iterator<It> mkRev(It it) {
  return std::reverse_iterator<It>(it);
}

void checkIteratorConcepts() {
  using namespace fs;
  using It = path::iterator;
  using Traits = std::iterator_traits<It>;
  ASSERT_SAME_TYPE(Traits::iterator_category, std::bidirectional_iterator_tag);
  ASSERT_SAME_TYPE(Traits::value_type, path);
  ASSERT_SAME_TYPE(Traits::pointer,   path const*);
  ASSERT_SAME_TYPE(Traits::reference, path const&);
  {
    It it;
    ASSERT_SAME_TYPE(It&, decltype(++it));
    ASSERT_SAME_TYPE(It, decltype(it++));
    ASSERT_SAME_TYPE(It&, decltype(--it));
    ASSERT_SAME_TYPE(It, decltype(it--));
    ASSERT_SAME_TYPE(Traits::reference, decltype(*it));
    ASSERT_SAME_TYPE(Traits::pointer, decltype(it.operator->()));
#ifdef _WIN32
    ASSERT_SAME_TYPE(std::wstring const&, decltype(it->native()));
#else
    ASSERT_SAME_TYPE(std::string const&, decltype(it->native()));
#endif
    ASSERT_SAME_TYPE(bool, decltype(it == it));
    ASSERT_SAME_TYPE(bool, decltype(it != it));
  }
  {
    path const p;
    ASSERT_SAME_TYPE(It, decltype(p.begin()));
    ASSERT_SAME_TYPE(It, decltype(p.end()));
    assert(p.begin() == p.end());
  }
}

void checkBeginEndBasic() {
  using namespace fs;
  using It = path::iterator;
  {
    path const p;
    ASSERT_SAME_TYPE(It, decltype(p.begin()));
    ASSERT_SAME_TYPE(It, decltype(p.end()));
    assert(p.begin() == p.end());
  }
  {
    path const p("foo");
    It default_constructed;
    default_constructed = p.begin();
    assert(default_constructed == p.begin());
    assert(default_constructed != p.end());
    default_constructed = p.end();
    assert(default_constructed == p.end());
    assert(default_constructed != p.begin());
  }
  {
    path p("//root_name//first_dir////second_dir");
#ifdef _WIN32
    const path expect[] = {"//root_name", "/", "first_dir", "second_dir"};
#else
    const path expect[] = {"/", "root_name", "first_dir", "second_dir"};
#endif
    assert(checkCollectionsEqual(p.begin(), p.end(), std::begin(expect), std::end(expect)));
    assert(checkCollectionsEqualBackwards(p.begin(), p.end(), std::begin(expect), std::end(expect)));

  }
  {
    path p("////foo/bar/baz///");
    const path expect[] = {"/", "foo", "bar", "baz", ""};
    assert(checkCollectionsEqual(p.begin(), p.end(), std::begin(expect), std::end(expect)));
    assert(checkCollectionsEqualBackwards(p.begin(), p.end(), std::begin(expect), std::end(expect)));

  }

}

int main(int, char**) {
  using namespace fs;
  checkIteratorConcepts();
  checkBeginEndBasic(); // See path.decompose.pass.cpp for more tests.

  return 0;
}
