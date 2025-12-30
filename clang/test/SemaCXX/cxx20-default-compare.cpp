// RUN: %clang_cc1 %s -std=c++23 -verify -Wfloat-equal

#include "Inputs/std-compare.h"

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

namespace GH102588 {
struct A {
  int i = 0;
  constexpr operator int() const { return i; }
  constexpr operator int&() { return ++i; }
};

struct B : A {
  bool operator==(const B &) const = default;
};

constexpr bool f() {
  B x;
  return x == x;
}

static_assert(f());

struct ConstOnly {
  std::strong_ordering operator<=>(const ConstOnly&) const;
  std::strong_ordering operator<=>(ConstOnly&) = delete;
  friend bool operator==(const ConstOnly&, const ConstOnly&);
  friend bool operator==(ConstOnly&, ConstOnly&) = delete;
};

struct MutOnly {
  std::strong_ordering operator<=>(const MutOnly&) const = delete;;
  std::strong_ordering operator<=>(MutOnly&);
  friend bool operator==(const MutOnly&, const MutOnly&) = delete;;
  friend bool operator==(MutOnly&, MutOnly&);
};

struct ConstCheck : ConstOnly {
  friend std::strong_ordering operator<=>(const ConstCheck&, const ConstCheck&) = default;
  std::strong_ordering operator<=>(ConstCheck const& __restrict) const __restrict = default;
  friend bool operator==(const ConstCheck&, const ConstCheck&) = default;
  bool operator==(this const ConstCheck&, const ConstCheck&) = default;
};

// FIXME: Non-reference explicit object parameter are rejected
struct MutCheck : MutOnly {
  friend bool operator==(MutCheck, MutCheck) = default;
  // std::strong_ordering operator<=>(this MutCheck, MutCheck) = default;
  friend std::strong_ordering operator<=>(MutCheck, MutCheck) = default;
  // bool operator==(this MutCheck, MutCheck) = default;
};
}
