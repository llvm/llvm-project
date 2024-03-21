// RUN: %clang_cc1 -verify %s -std=c++11

template<typename T>
struct S0 {
  enum E0 : int;

  enum class E1;
};

struct S3 {
  enum E2 : int;

  enum class E3;
};

template<typename T>
enum S0<T>::E0 : int; // expected-error{{cannot have a nested name specifier}}

template<>
enum S0<int>::E0 : int;

template<typename T>
enum class S0<T>::E1; // expected-error{{cannot have a nested name specifier}}

template<>
enum class S0<int>::E1;

enum S3::E2 : int; // expected-error{{cannot have a nested name specifier}}

enum class S3::E3; // expected-error{{cannot have a nested name specifier}}
