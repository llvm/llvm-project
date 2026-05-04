//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

// class regex_iterator<BidirectionalIterator, charT, traits>

// regex_iterator operator++(int);

#include <regex>
#include <cassert>
#include <iterator>
#include "test_macros.h"

void validate_prefixes(const std::regex& empty_matching_pattern) {
  const char source[] = "abc";

  std::cregex_iterator i(source, source + 3, empty_matching_pattern);
  assert(!i->prefix().matched);
  assert(i->prefix().length() == 0);
  assert(i->prefix().first == source);
  assert(i->prefix().second == source);

  ++i;
  assert(i->prefix().matched);
  assert(i->prefix().length() == 1);
  assert(i->prefix().first == source);
  assert(i->prefix().second == source + 1);
  assert(i->prefix().str() == "a");

  ++i;
  assert(i->prefix().matched);
  assert(i->prefix().length() == 1);
  assert(i->prefix().first == source + 1);
  assert(i->prefix().second == source + 2);
  assert(i->prefix().str() == "b");

  ++i;
  assert(i->prefix().matched);
  assert(i->prefix().length() == 1);
  assert(i->prefix().first == source + 2);
  assert(i->prefix().second == source + 3);
  assert(i->prefix().str() == "c");

  ++i;
  assert(i == std::cregex_iterator());
}

void test_prefix_adjustment() {
  // Check that we correctly adjust the match prefix when dealing with zero-length matches -- this is explicitly
  // required by the Standard ([re.regiter.incr]: "In all cases in which the call to `regex_search` returns true,
  // `match.prefix().first` shall be equal to the previous value of `match[0].second`"). For a pattern that matches
  // empty sequences, there is an implicit zero-length match between every character in a string -- make sure the
  // prefix of each of these matches (except the first one) is the preceding character.

  // An empty pattern produces zero-length matches.
  validate_prefixes(std::regex(""));
  // Any character repeated zero or more times can produce zero-length matches.
  validate_prefixes(std::regex("X*"));
  validate_prefixes(std::regex("X{0,3}"));
}

int main(int, char**) {
  {
    std::regex phone_numbers("\\d{3}-\\d{4}");
    const char phone_book[] = "555-1234, 555-2345, 555-3456";
    std::cregex_iterator i(std::begin(phone_book), std::end(phone_book), phone_numbers);
    std::cregex_iterator i2 = i;
    assert(i != std::cregex_iterator());
    assert(i2 != std::cregex_iterator());
    assert((*i).size() == 1);
    assert((*i).position() == 0);
    assert((*i).str() == "555-1234");
    assert((*i2).size() == 1);
    assert((*i2).position() == 0);
    assert((*i2).str() == "555-1234");
    i++;
    assert(i != std::cregex_iterator());
    assert(i2 != std::cregex_iterator());
    assert((*i).size() == 1);
    assert((*i).position() == 10);
    assert((*i).str() == "555-2345");
    assert((*i2).size() == 1);
    assert((*i2).position() == 0);
    assert((*i2).str() == "555-1234");
    i++;
    assert(i != std::cregex_iterator());
    assert(i2 != std::cregex_iterator());
    assert((*i).size() == 1);
    assert((*i).position() == 20);
    assert((*i).str() == "555-3456");
    assert((*i2).size() == 1);
    assert((*i2).position() == 0);
    assert((*i2).str() == "555-1234");
    i++;
    assert(i == std::cregex_iterator());
    assert(i2 != std::cregex_iterator());
    assert((*i2).size() == 1);
    assert((*i2).position() == 0);
    assert((*i2).str() == "555-1234");
  }
  {
    std::regex phone_numbers("\\d{3}-\\d{4}");
    const char phone_book[] = "555-1234, 555-2345, 555-3456";
    std::cregex_iterator i(std::begin(phone_book), std::end(phone_book), phone_numbers);
    std::cregex_iterator i2 = i;
    assert(i != std::cregex_iterator());
    assert(i2 != std::cregex_iterator());
    assert((*i).size() == 1);
    assert((*i).position() == 0);
    assert((*i).str() == "555-1234");
    assert((*i2).size() == 1);
    assert((*i2).position() == 0);
    assert((*i2).str() == "555-1234");
    ++i;
    assert(i != std::cregex_iterator());
    assert(i2 != std::cregex_iterator());
    assert((*i).size() == 1);
    assert((*i).position() == 10);
    assert((*i).str() == "555-2345");
    assert((*i2).size() == 1);
    assert((*i2).position() == 0);
    assert((*i2).str() == "555-1234");
    ++i;
    assert(i != std::cregex_iterator());
    assert(i2 != std::cregex_iterator());
    assert((*i).size() == 1);
    assert((*i).position() == 20);
    assert((*i).str() == "555-3456");
    assert((*i2).size() == 1);
    assert((*i2).position() == 0);
    assert((*i2).str() == "555-1234");
    ++i;
    assert(i == std::cregex_iterator());
    assert(i2 != std::cregex_iterator());
    assert((*i2).size() == 1);
    assert((*i2).position() == 0);
    assert((*i2).str() == "555-1234");
  }
  { // https://llvm.org/PR33681
    std::regex rex(".*");
    const char foo[] = "foo";
    //  The -1 is because we don't want the implicit null from the array.
    std::cregex_iterator i(std::begin(foo), std::end(foo) - 1, rex);
    std::cregex_iterator e;
    assert(i != e);
    assert((*i).size() == 1);
    assert((*i).str() == "foo");

    ++i;
    assert(i != e);
    assert((*i).size() == 1);
    assert((*i).str() == "");

    ++i;
    assert(i == e);
  }

  test_prefix_adjustment();

  return 0;
}
