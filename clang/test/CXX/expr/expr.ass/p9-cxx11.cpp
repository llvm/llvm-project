// RUN: %clang_cc1 -verify -std=c++11 %s

template<typename T> struct complex {
  complex(T = T(), T = T());
  void operator+=(complex);
  T a, b;
};

void std_example() {
  complex<double> z;
  z = { 1, 2 };
  z += { 1, 2 };

  int a, b;
  a = b = { 1 };
  a = { 1 } = b; // expected-error {{initializer list cannot be used on the left hand side of operator '='}}
  a = a + { 4 }; // expected-error {{initializer list cannot be used on the right hand side of operator '+'}}
  a = { 3 } * { 4 }; // expected-error {{initializer list cannot be used on the left hand side of operator '*'}} \
                        expected-error {{initializer list cannot be used on the right hand side of operator '*'}}
}

struct S {
  constexpr S(int a, int b) : a(a), b(b) {}
  int a, b;
};
struct T {
  constexpr int operator=(S s) const { return s.a; }
  constexpr int operator+=(S s) const { return s.b; }
};
static_assert((T() = {4, 9}) == 4, "");
static_assert((T() += {4, 9}) == 9, "");

int k1 = T() = { 1, 2 } = { 3, 4 }; // expected-error {{initializer list cannot be used on the left hand side of operator '='}}
int k2 = T() = { 1, 2 } + 1; // expected-error {{initializer list cannot be used on the left hand side of operator '+'}}
