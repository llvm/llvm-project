//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// class num_put<charT, OutputIterator>

// LWG4084: std::fixed ignores std::uppercase

#include <cassert>
#include <ios>
#include <limits>
#include <locale>
#include <string>

#include "test_iterators.h"

typedef std::num_put<char, cpp17_output_iterator<char*> > Facet;

class my_facet : public Facet {
public:
  explicit my_facet(std::size_t refs = 0) : Facet(refs) {}
};

template <class T>
std::string put(T value, std::ios_base& ios) {
  char str[100];
  const my_facet f(1);
  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', value);
  return std::string(str, base(iter));
}

bool is_lowercase_infinity(const std::string& str) { return str == "inf" || str == "infinity"; }

bool is_uppercase_infinity(const std::string& str) { return str == "INF" || str == "INFINITY"; }

template <class T>
void test() {
  {
    std::ios ios(0);
    std::fixed(ios);
    std::nouppercase(ios);

    assert(is_lowercase_infinity(put(std::numeric_limits<T>::infinity(), ios)));
  }
  {
    std::ios ios(0);
    std::fixed(ios);
    std::uppercase(ios);

    assert(is_uppercase_infinity(put(std::numeric_limits<T>::infinity(), ios)));
  }
}

int main(int, char**) {
  test<double>();
  test<long double>();

  return 0;
}
