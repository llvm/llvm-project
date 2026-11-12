// RUN: %clang_cc1 -std=c++98 %s -fsyntax-only
// RUN: %clang_cc1 -std=c++11 %s -verify

// In C++11, we must perform overload resolution to determine which function is
// called by a defaulted assignment operator, and the selected operator might
// not be a copy or move assignment (it might be a specialization of a templated
// 'operator=', for instance).
struct A {
  A &operator=(const A &);

  template<typename T>
  A &operator=(T &&) { return T::error; } // expected-error {{no member named 'error' in 'A'}}
};

struct B : A {
  B &operator=(B &&);
};

B &B::operator=(B &&) = default; // expected-note {{here}}
