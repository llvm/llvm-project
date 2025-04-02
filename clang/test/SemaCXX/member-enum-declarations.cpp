// RUN: %clang_cc1 -std=c++11 -fsyntax-only %s -verify
// RUN: %clang_cc1 -std=c++14 -fsyntax-only %s -verify
// RUN: %clang_cc1 -std=c++20 -fsyntax-only %s -verify


namespace ScopedEnumerations {

template <typename T>
struct S1 {
  enum class E : T;
};

template <typename T>
enum class S1<T>::E : T {
  S1_X = 0x123
};

static_assert(static_cast<int>(S1<int>::E::S1_X) == 0x123, "");

template <typename T>
struct S2 {
  static constexpr T f(int) { return 0; };
  enum class E : T;
  static constexpr T f(char) { return 1; };
  enum class E : T { X = f(T{}) };
};

static_assert(static_cast<int>(S2<char>::E::X) == 1, "");

template <typename T>
struct S3 {
  enum class E : T;
  enum class E : T { X = 0x7FFFFF00 }; // expected-error {{cannot be narrowed to type 'char'}} expected-warning {{implicit conversion from 'int' to 'char'}}
};
template struct S3<char>; // expected-note {{in instantiation}}

template <typename T>
struct S4 {
  enum class E : T;
  enum class E : T { S4_X = 5 };
};

auto x4 = S4<int>::E::S4_X;

template <typename T>
T f1() {
  enum class E : T { X_F1, Y_F1, Z_F1 };
  return X_F1;  // expected-error {{use of undeclared identifier 'X_F1'}}
}

const int resf1 = f1<int>();

}


namespace UnscopedEnumerations {

template <typename T>
struct S1 {
  enum E : T;
};

template <typename T>
enum S1<T>::E : T {
  S1_X = 0x123
};

static_assert(static_cast<int>(S1<int>::S1_X) == 0x123, "");

template <typename T>
struct S2 {
  static constexpr T f(int) { return 0; };
  enum E : T;
  static constexpr T f(char) { return 1; };
  enum E : T { S2_X = f(T{}) };
};

static_assert(static_cast<int>(S2<char>::E::S2_X) == 1, "");

template <typename T>
struct S3 {
  enum E : T;
  enum E : T { S3_X = 0x7FFFFF00 }; // expected-error {{cannot be narrowed to type 'char'}} expected-warning {{implicit conversion from 'int' to 'char'}}
};
template struct S3<char>; // expected-note {{in instantiation of template class}}

template <typename T>
struct S4 {
  enum E : T;
  enum E : T { S4_X = 5 };
};

auto x4 = S4<int>::S4_X;

template <typename T>
struct S5 {
  enum E : T;
  T S5_X = 5; // expected-note {{previous definition is here}}
  enum E : T { S5_X = 5 }; // expected-error {{redefinition of 'S5_X'}}
};


template <typename T>
T f1() {
  enum E : T { X_F2, Y_F2, Z_F2 };
  return X_F2;
}

const int resf1 = f1<int>();

}

