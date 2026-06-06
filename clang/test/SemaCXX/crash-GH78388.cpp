// RUN: %clang_cc1 -verify -std=c++98 %s
// RUN: %clang_cc1 -verify -std=c++11 %s
// RUN: %clang_cc1 -verify -std=c++14 %s
// RUN: %clang_cc1 -verify -std=c++17 %s
// RUN: %clang_cc1 -verify -std=c++20 %s
// RUN: %clang_cc1 -verify -std=c++23 %s
// RUN: %clang_cc1 -verify -std=c++2c %s

// https://github.com/llvm/llvm-project/issues/78388

typedef mbstate_t; // expected-error 0+ {{}} expected-note 0+ {{}}
  template < typename , typename , typename >
  class a // expected-error 0+ {{}}
  class b { // expected-error 0+ {{}}
    namespace { // expected-note 0+ {{}} expected-note 0+ {{}}
    template < typename c > b::operator=() { // expected-error 0+ {{}} expected-note 0+ {{}}
      struct :a< c, char, stdmbstate_t > d // expected-error 0+ {{}} expected-warning 0+ {{}}
