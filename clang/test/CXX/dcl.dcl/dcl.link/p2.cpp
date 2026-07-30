// RUN: %clang_cc1 -std=c++11 -verify %s

extern "C" {
  extern R"(C++)" { }
}

#define plusplus "++"
extern "C" plusplus {
}

extern u8"C" {}  // expected-warning {{encoding prefix 'u8' on an unevaluated string literal has no effect and is incompatible with c++2c}}
extern L"C" {}   // expected-warning {{encoding prefix 'L' on an unevaluated string literal has no effect and is incompatible with c++2c}}
extern u"C++" {} // expected-warning {{encoding prefix 'u' on an unevaluated string literal has no effect and is incompatible with c++2c}}
extern U"C" {}   // expected-warning {{encoding prefix 'U' on an unevaluated string literal has no effect and is incompatible with c++2c}}
