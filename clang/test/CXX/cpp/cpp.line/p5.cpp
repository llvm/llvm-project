// RUN: %clang_cc1 -std=c++26 %s -fsyntax-only -verify

// [cpp.line]/p5: If the directive resulting after all replacements does not
// match one of the two previous forms, the program is ill-formed.
#line -1 // expected-error {{#line directive requires a positive integer argument}}
