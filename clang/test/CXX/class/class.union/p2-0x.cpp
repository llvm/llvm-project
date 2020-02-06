// RUN: %clang_cc1 -verify -std=c++11 %s

// Unlike in C++98, C++11 allows unions to have static data members.

union U1 {
  static constexpr int k1 = 0;
  static const int k2 = k1;
  static int k3 = k2; // expected-error {{non-const static data member must be initialized out of line}}
  static constexpr double k4 = k2;
  static const double k5 = k4; // expected-error {{requires 'constexpr' specifier}} expected-note {{add 'constexpr'}}
  int n[k1 + 3];
};

constexpr int U1::k1;
constexpr int U1::k2;
int U1::k3;

const double U1::k4;
const double U1::k5;

template<typename T>
union U2 {
  static const int k1;
  static double k2;
  T t;
};
template<typename T> constexpr int U2<T>::k1 = sizeof(U2<T>);
template<typename T> double U2<T>::k2 = 5.3;

static_assert(U2<int>::k1 == sizeof(int), "");
static_assert(U2<char>::k1 == sizeof(char), "");

union U3 {
  static const int k;
  U3() : k(0) {} // expected-error {{does not name a non-static data member}}
};

struct S {
  union {
    static const int n; // expected-error {{static data member 'n' not allowed in anonymous union}}
    int a;
    int b;
  };
};
static union {
  static const int k; // expected-error {{static data member 'k' not allowed in anonymous union}}
  int n;
};
