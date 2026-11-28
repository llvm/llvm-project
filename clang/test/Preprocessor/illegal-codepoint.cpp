// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

// https://github.com/llvm/llvm-project/issues/125461

#define Y\U00110000Z() "Y\U00110000Z is an identifier" // expected-warning {{ISO C99 requires whitespace after the macro name}} \
                                                       // expected-note {{previous definition is here}}
#define Y\U00110001Z() "Y\U00110001Z is an identifier" // expected-warning {{ISO C99 requires whitespace after the macro name}} \
                                                       // expected-warning {{'Y' macro redefined}}
Y\U00110000Z() // expected-error {{character <0x110000> not allowed in an identifier}}
YZ() // expected-error {{expected function body after function declarator}}
