//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// Check that functions are marked [[nodiscard]]

#include <cwchar>
#include <locale>
#include <string>

#include "test_macros.h"

template <class Facet>
struct derived_facet : Facet {
  derived_facet() {}
};

void test() {
  // [locales]
  {
    std::locale l;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    l.combine<std::messages<char> >(l);
    l.name(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    l(std::string(), std::string());

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::locale::classic();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::use_facet<std::messages<char> >(l);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::has_facet<std::messages<char> >(l);

    std::isspace('\r', l); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::isprint('*', l);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::iscntrl('\n', l); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::isupper('A', l);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::islower('b', l);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::isalpha('C', l);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::isdigit('0', l);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ispunct(',', l);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::isxdigit('d', l); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::isalnum('Z', l);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::isgraph('!', l);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::isblank(' ', l);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    std::toupper('g', l); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::tolower('H', l); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  // [locale.ctype.general]
  {
    derived_facet<std::ctype<wchar_t> > f;

    f.is(0, L'\0'); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.scan_is(0, nullptr, nullptr);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.scan_not(0, nullptr, nullptr);

    f.toupper(L'a'); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.tolower(L'B'); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    f.widen('c');         // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.narrow(L'D', L'*'); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
#endif
  // [facet.ctype.special.general]
  {
    derived_facet<std::ctype<char> > f;

    f.is(0, '\0'); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.scan_is(0, nullptr, nullptr);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.scan_not(0, nullptr, nullptr);

    f.toupper('a'); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.tolower('B'); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    f.widen('c');       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.narrow('D', '*'); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    f.table(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ctype<char>::classic_table();
  }

  // [locale.codecvt.general]
  {
    derived_facet<std::codecvt<char, char, std::mbstate_t> > f;

    f.encoding();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.always_noconv(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.max_length();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  {
    derived_facet<std::codecvt<wchar_t, char, std::mbstate_t> > f;

    f.encoding();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.always_noconv(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.max_length();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
#endif
  {
    derived_facet<std::codecvt<char16_t, char, std::mbstate_t> > f;

    f.encoding();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.always_noconv(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.max_length();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
  {
    derived_facet<std::codecvt<char32_t, char, std::mbstate_t> > f;

    f.encoding();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.always_noconv(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.max_length();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
#ifndef TEST_HAS_NO_CHAR8_T
  {
    derived_facet<std::codecvt<char16_t, char8_t, std::mbstate_t> > f;

    f.encoding();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.always_noconv(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.max_length();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
  {
    derived_facet<std::codecvt<char32_t, char8_t, std::mbstate_t> > f;

    f.encoding();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.always_noconv(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.max_length();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
#endif

  // [locale.numpunct.general]
  {
    derived_facet<std::numpunct<char> > f;

    f.decimal_point(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.thousands_sep(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.grouping();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.truename();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.falsename();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  {
    derived_facet<std::numpunct<wchar_t> > f;

    f.decimal_point(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.thousands_sep(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.grouping();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.truename();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.falsename();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
#endif

  // [locale.collate.general]
  {
    derived_facet<std::collate<char> > f;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.compare(nullptr, nullptr, nullptr, nullptr);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.transform(nullptr, nullptr);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.hash(nullptr, nullptr);
  }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  {
    derived_facet<std::collate<wchar_t> > f;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.compare(nullptr, nullptr, nullptr, nullptr);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.transform(nullptr, nullptr);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.hash(nullptr, nullptr);
  }
#endif

  // [locale.moneypunct.general]
  {
    derived_facet<std::moneypunct<char> > f;

    f.decimal_point(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.thousands_sep(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.grouping();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.curr_symbol();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.positive_sign(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.negative_sign(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.frac_digits();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.pos_format();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.neg_format();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  {
    derived_facet<std::moneypunct<wchar_t> > f;

    f.decimal_point(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.thousands_sep(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.grouping();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.curr_symbol();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.positive_sign(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.negative_sign(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.frac_digits();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.pos_format();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.neg_format();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
#endif

  // [locale.messages.general]
  {
    derived_facet<std::messages<char> > f;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.open(std::string(), std::locale());
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.get(0, 0, 0, std::string());
  }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  {
    derived_facet<std::messages<wchar_t> > f;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.open(std::string(), std::locale());
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    f.get(0, 0, 0, std::wstring());
  }
#endif
}
