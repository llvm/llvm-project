//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-localization

// check that <regex> functions are marked [[nodiscard]]

#include <regex>
#include <string>

void test() {
  {
    std::basic_regex<char> re;

    re.mark_count(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    re.flags();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    re.getloc(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
  {
    std::sub_match<const char*> sm;

    sm.length(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    sm.str();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    sm.compare(sm);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    sm.compare(std::string());
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    sm.compare("");
  }
  {
    std::match_results<const char*> m;

    m.size();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    m.max_size(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    m.empty();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    m.length();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    m.position(0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    m.str();       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    m[0];          // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    m.prefix(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    m.suffix(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    m.begin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    m.end();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    m.cbegin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    m.cend();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    m.format(std::string());
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    m.format("");
  }
  {
    std::regex_iterator<const char*> ri;

    *ri; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
  {
    std::regex_token_iterator<const char*> rti;

    *rti; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
  {
    std::regex_error err(std::regex_constants::error_backref);

    err.code(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
}
