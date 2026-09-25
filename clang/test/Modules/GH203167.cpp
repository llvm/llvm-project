// RUN: %clang_cc1 -std=c++20 -fmodules -fsyntax-only -verify %s

#pragma clang module build N // expected-error {{no matching '#pragma clang module endbuild'}}
module N {}
#pragma clang module contents
#pragma clang module begin N // expected-error {{no matching '#pragma clang module end'}}
int x;
