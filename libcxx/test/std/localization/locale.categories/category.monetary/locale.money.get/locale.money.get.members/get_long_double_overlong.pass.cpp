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

// Ensure that money_get::do_get correct works when the input doesn't fit into the stack buffer
// (100 characters currently).

#include <cassert>
#include <cstddef>
#include <ios>
#include <locale>
#include <streambuf>
#include <string>

#include "make_string.h"
#include "test_macros.h"
#include "test_iterators.h"

template <class CharT>
class my_basic_facet : public std::money_get<CharT, cpp17_input_iterator<const CharT*> > {
private:
  typedef std::money_get<CharT, cpp17_input_iterator<const CharT*> > Base;

public:
  explicit my_basic_facet(std::size_t refs = 0) : Base(refs) {}
};

template <class CharT>
void test() {
  struct digit_result_case {
    std::size_t digit;
    long double result;
  };
  const digit_result_case digit_result_cases[] = {
      {60, 2.0E60L}, {120, 2.0E120L}, {180, 2.0E180L}, {240, 2.0E240L}, {300, 2.0E300L}};

  std::ios ios(0);
  {
    const my_basic_facet<CharT> f(1);
    for (std::size_t i = 0; i != sizeof(digit_result_cases) / sizeof(digit_result_cases[0]); ++i) {
      {
        std::basic_string<CharT> v = MAKE_STRING(CharT, "2");
        v.append(digit_result_cases[i].digit, static_cast<CharT>('0'));

        typedef cpp17_input_iterator<const CharT*> I;
        long double ex;
        std::ios_base::iostate err = std::ios_base::goodbit;
        I iter                     = f.get(I(v.data()), I(v.data() + v.size()), false, ios, err, ex);
        assert(base(iter) == v.data() + v.size());
        assert(err == std::ios_base::eofbit);
        assert(ex == digit_result_cases[i].result);
      }
      {
        std::basic_string<CharT> v = MAKE_STRING(CharT, "-2");
        v.append(digit_result_cases[i].digit, static_cast<CharT>('0'));

        typedef cpp17_input_iterator<const CharT*> I;
        long double ex;
        std::ios_base::iostate err = std::ios_base::goodbit;
        I iter                     = f.get(I(v.data()), I(v.data() + v.size()), false, ios, err, ex);
        assert(base(iter) == v.data() + v.size());
        assert(err == std::ios_base::eofbit);
        assert(ex == -digit_result_cases[i].result);
      }
      {
        std::basic_string<CharT> v = MAKE_STRING(CharT, "0.");
        v.append(digit_result_cases[i].digit, static_cast<CharT>('0'));
        v += MAKE_CSTRING(CharT, "2");

        typedef cpp17_input_iterator<const CharT*> I;
        long double ex;
        std::ios_base::iostate err = std::ios_base::goodbit;
        I iter                     = f.get(I(v.data()), I(v.data() + v.size()), false, ios, err, ex);
        assert(base(iter) == v.data() + 1);
        assert(err == std::ios_base::goodbit);
        assert(ex == 0.0L);
      }
      {
        std::basic_string<CharT> v = MAKE_STRING(CharT, "-0.");
        v.append(digit_result_cases[i].digit, static_cast<CharT>('0'));
        v += MAKE_CSTRING(CharT, "2");

        typedef cpp17_input_iterator<const CharT*> I;
        long double ex;
        std::ios_base::iostate err = std::ios_base::goodbit;
        I iter                     = f.get(I(v.data()), I(v.data() + v.size()), false, ios, err, ex);
        assert(base(iter) == v.data() + 2);
        assert(err == std::ios_base::goodbit);
        assert(ex == 0.0L);
      }
    }
  }
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
