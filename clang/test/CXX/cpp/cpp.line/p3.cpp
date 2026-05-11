// RUN: %clang_cc1 -std=c++26 %s -fsyntax-only -verify

// [cpp.line]/p3: Digit sequences representing a number outside of the range
// [1, 2147483647] are conditionally supported.
#line 0 // expected-warning {{#line directive with zero argument is a GNU extension}}
#line 2147483647 // ok, largest value required to be accepted
#line 2147483648 // expected-warning {{C requires #line number to be less than 2147483648, allowed as extension}}
