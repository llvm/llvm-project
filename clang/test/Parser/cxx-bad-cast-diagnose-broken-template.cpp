// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fsyntax-only -verify %s

template< typename>
struct IsConstCharArray
{
  static const bool value = false;
};

template< int N >
struct IsConstCharArray< const char[ N ] >
{
  typedef char CharType;
  static const bool value = true;
  static const missing_int_t length = N - 1; // expected-error {{unknown type name 'missing_int_t'}}
};

class String {
public:
  template <typename T>
  String(T& str, typename IsConstCharArray<T>::CharType = 0);
};


class Exception {
public:
  Exception(String const&);
};

void foo() {
  throw Exception("some error"); // expected-error {{functional-style cast from 'const char[11]' to 'Exception' is not allowed}}
}
