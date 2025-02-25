//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// class money_get<charT, InputIterator>

// iter_type get(iter_type b, iter_type e, bool intl, ios_base& iob,
//               ios_base::iostate& err, long double& v) const;

#include <cassert>
#include <cstddef>
#include <ios>
#include <locale>
#include <streambuf>
#include <string>

#include "test_macros.h"
#include "test_iterators.h"

typedef std::money_get<char, cpp17_input_iterator<const char*> > Fn;

class my_facet : public Fn {
public:
  explicit my_facet(std::size_t refs = 0) : Fn(refs) {}
};

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
typedef std::money_get<wchar_t, cpp17_input_iterator<const wchar_t*> > Fw;

class my_facetw : public Fw {
public:
  explicit my_facetw(std::size_t refs = 0) : Fw(refs) {}
};
#endif

int main(int, char**) {
  struct digit_result_case {
    std::size_t digit;
    long double result;
  };
  const digit_result_case digit_result_cases[] = {
      {60, 2.0E60L}, {120, 2.0E120L}, {180, 2.0E180L}, {240, 2.0E240L}, {300, 2.0E300L}};

  std::ios ios(0);
  {
    const my_facet f(1);
    for (std::size_t i = 0; i != sizeof(digit_result_cases) / sizeof(digit_result_cases[0]); ++i) {
      {
        std::string v = "2";
        v.append(digit_result_cases[i].digit, '0');

        typedef cpp17_input_iterator<const char*> I;
        long double ex;
        std::ios_base::iostate err = std::ios_base::goodbit;
        I iter                     = f.get(I(v.data()), I(v.data() + v.size()), false, ios, err, ex);
        assert(base(iter) == v.data() + v.size());
        assert(err == std::ios_base::eofbit);
        assert(ex == digit_result_cases[i].result);
      }
      {
        std::string v = "-2";
        v.append(digit_result_cases[i].digit, '0');

        typedef cpp17_input_iterator<const char*> I;
        long double ex;
        std::ios_base::iostate err = std::ios_base::goodbit;
        I iter                     = f.get(I(v.data()), I(v.data() + v.size()), false, ios, err, ex);
        assert(base(iter) == v.data() + v.size());
        assert(err == std::ios_base::eofbit);
        assert(ex == -digit_result_cases[i].result);
      }
    }
  }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  {
    const my_facetw f(1);
    for (std::size_t i = 0; i != sizeof(digit_result_cases) / sizeof(digit_result_cases[0]); ++i) {
      {
        std::wstring v = L"2";
        v.append(digit_result_cases[i].digit, L'0');

        typedef cpp17_input_iterator<const wchar_t*> I;
        long double ex;
        std::ios_base::iostate err = std::ios_base::goodbit;
        I iter                     = f.get(I(v.data()), I(v.data() + v.size()), false, ios, err, ex);
        assert(base(iter) == v.data() + v.size());
        assert(err == std::ios_base::eofbit);
        assert(ex == digit_result_cases[i].result);
      }
      {
        std::wstring v = L"-2";
        v.append(digit_result_cases[i].digit, L'0');

        typedef cpp17_input_iterator<const wchar_t*> I;
        long double ex;
        std::ios_base::iostate err = std::ios_base::goodbit;
        I iter                     = f.get(I(v.data()), I(v.data() + v.size()), false, ios, err, ex);
        assert(base(iter) == v.data() + v.size());
        assert(err == std::ios_base::eofbit);
        assert(ex == -digit_result_cases[i].result);
      }
    }
  }
#endif

  return 0;
}
