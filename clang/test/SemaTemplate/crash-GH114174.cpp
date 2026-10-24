// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

template <typename T>
class X {
public:
  // expected-error@+1 {{in-class initializer for static data member is not a constant expression}}
  static const T value = 0 / 0;
};

// expected-note@+2 {{in instantiation of template class 'X<int>' requested here}}
// expected-error@+1 {{use of undeclared identifier 'y'}}
X<int> x = y;

namespace std {
class type_info {};
}

template <typename T>
struct Y {
  static const int value = sizeof(typeid(T));
  const int value_1 = value * 1;
};
