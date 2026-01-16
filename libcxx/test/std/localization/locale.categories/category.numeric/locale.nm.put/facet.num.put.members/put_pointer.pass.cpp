//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// class num_put<charT, OutputIterator>

// iter_type put(iter_type s, ios_base& iob, char_type fill, void* v) const;

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <cassert>
#include <ios>
#include <locale>

#include "test_iterators.h"

typedef std::num_put<char, cpp17_output_iterator<char*> > F;

class my_facet : public F {
public:
  explicit my_facet(std::size_t refs = 0) : F(refs) {}
};

int main(int, char**) {
  const my_facet f(1);
  {
    std::ios ios(nullptr);
    void* v = nullptr;
    char str[50];
    cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
    std::string ex(str, base(iter));
    assert(!ex.empty());
    LIBCPP_NON_FROZEN_ASSERT(ex == "0");
  }

  return 0;
}
