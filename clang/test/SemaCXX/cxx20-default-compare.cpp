// RUN: %clang_cc1 %s -std=c++23 -verify -Wfloat-equal

struct Foo {
  float val;
  bool operator==(const Foo &) const;
  friend bool operator==(const Foo &, const Foo &);
  friend bool operator==(Foo, Foo );
};

// Declare the defaulted comparison function as a member function.
bool Foo::operator==(const Foo &) const = default; // expected-warning {{comparing floating point with == or != is unsafe}} expected-note {{in defaulted equality comparison operator for 'Foo' first required here}}

// Declare the defaulted comparison function as a non-member function.
bool operator==(const Foo &, const Foo &) = default;  // expected-warning {{comparing floating point with == or != is unsafe}} expected-note {{in defaulted equality comparison operator for 'Foo' first required here}}

// Declare the defaulted comparison function as a non-member function. Arguments are passed by value.
bool operator==(Foo, Foo) = default;  // expected-warning {{comparing floating point with == or != is unsafe}} expected-note {{in defaulted equality comparison operator for 'Foo' first required here}}
