// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fsyntax-only -verify %s

template<typename>
struct StringTrait {};

template< int N >
struct StringTrait< const char[ N ] > {
  typedef char CharType;
  static const MissingIntT length = N - 1; // expected-error {{unknown type name 'MissingIntT'}}
};

class String {
public:
  template <typename T>
  String(T& str, typename StringTrait<T>::CharType = 0);
};


class Exception {
public:
  Exception(String const&);
};

void foo() {
  throw Exception("some error"); // expected-error {{functional-style cast from 'const char[11]' to 'Exception' is not allowed}}
}
