// RUN: %clang_cc1 %s -std=c++23 -verify -Wfloat-equal

struct Foo {
  float val;
  bool operator==(const Foo &) const;
  friend bool operator==(const Foo &, const Foo &);
  friend bool operator==(Foo, Foo );
};

struct X {};
namespace NS {
  bool operator==(X, X);
}
using namespace NS;

struct Y {
  X x;
  friend bool operator==(Y, Y);
};

template <typename T>
struct Z {
  T x;
  friend bool operator==(Z, Z);
};
template class Z<X>;

// Declare the defaulted comparison function as a member function.
bool Foo::operator==(const Foo &) const = default; // expected-warning {{comparing floating point with == or != is unsafe}} expected-note {{in defaulted equality comparison operator for 'Foo' first required here}}

// Declare the defaulted comparison function as a non-member function.
bool operator==(const Foo &, const Foo &) = default;  // expected-warning {{comparing floating point with == or != is unsafe}} expected-note {{in defaulted equality comparison operator for 'Foo' first required here}}

// Declare the defaulted comparison function as a non-member function. Arguments are passed by value.
bool operator==(Foo, Foo) = default;  // expected-warning {{comparing floating point with == or != is unsafe}} expected-note {{in defaulted equality comparison operator for 'Foo' first required here}}

// Declare the defaulted comparsion function as a non-member function. Arguments are passed by value. Arguments look up NS namespace.
bool operator==(Y, Y) = default; 

// Declare the defaulted comparsion function as a non-member function. Arguments are passed by value. Arguments look up NS namespace and use template struct.
bool operator==(Z<X>, Z<X>) = default;
