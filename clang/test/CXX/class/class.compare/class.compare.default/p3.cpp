// This test is for the [class.compare.default]p3 added by P2002R0
// Also covers modifications made by P2448R2 and extension warnings

// RUN: %clang_cc1 -std=c++2a -verify %s
// RUN: %clang_cc1 -std=c++2a -Wc++23-default-comp-relaxed-constexpr -verify=expected,extension %s

namespace std {
  struct strong_ordering {
    int n;
    constexpr operator int() const { return n; }
    static const strong_ordering less, equal, greater;
  };
  constexpr strong_ordering strong_ordering::less = {-1};
  constexpr strong_ordering strong_ordering::equal = {0};
  constexpr strong_ordering strong_ordering::greater = {1};
}

struct A {
  friend bool operator==(const A&, const A&) = default;
  friend bool operator!=(const A&, const A&) = default;

  friend std::strong_ordering operator<=>(const A&, const A&) = default;
  friend bool operator<(const A&, const A&) = default;
  friend bool operator<=(const A&, const A&) = default;
  friend bool operator>(const A&, const A&) = default;
  friend bool operator>=(const A&, const A&) = default;
};
struct TestA {
  friend constexpr bool operator==(const A&, const A&) noexcept;
  friend constexpr bool operator!=(const A&, const A&) noexcept;

  friend constexpr std::strong_ordering operator<=>(const A&, const A&) noexcept;
  friend constexpr bool operator<(const A&, const A&);
  friend constexpr bool operator<=(const A&, const A&);
  friend constexpr bool operator>(const A&, const A&);
  friend constexpr bool operator>=(const A&, const A&);
};

// Declaration order doesn't matter, even though the secondary operators need
// to know whether the primary ones are constexpr.
struct ReversedA {
  friend bool operator>=(const ReversedA&, const ReversedA&) = default;
  friend bool operator>(const ReversedA&, const ReversedA&) = default;
  friend bool operator<=(const ReversedA&, const ReversedA&) = default;
  friend bool operator<(const ReversedA&, const ReversedA&) = default;
  friend std::strong_ordering operator<=>(const ReversedA&, const ReversedA&) = default;

  friend bool operator!=(const ReversedA&, const ReversedA&) = default;
  friend bool operator==(const ReversedA&, const ReversedA&) = default;
};
struct TestReversedA {
  friend constexpr bool operator>=(const ReversedA&, const ReversedA&);
  friend constexpr bool operator>(const ReversedA&, const ReversedA&);
  friend constexpr bool operator<=(const ReversedA&, const ReversedA&);
  friend constexpr bool operator<(const ReversedA&, const ReversedA&);
  friend constexpr std::strong_ordering operator<=>(const ReversedA&, const ReversedA&) noexcept;

  friend constexpr bool operator!=(const ReversedA&, const ReversedA&) noexcept;
  friend constexpr bool operator==(const ReversedA&, const ReversedA&) noexcept;
};

struct B {
  A a;
  friend bool operator==(const B&, const B&) = default;
  friend bool operator!=(const B&, const B&) = default;

  friend std::strong_ordering operator<=>(const B&, const B&) = default;
  friend bool operator<(const B&, const B&) = default;
  friend bool operator<=(const B&, const B&) = default;
  friend bool operator>(const B&, const B&) = default;
  friend bool operator>=(const B&, const B&) = default;
};
struct TestB {
  friend constexpr bool operator==(const B&, const B&) noexcept;
  friend constexpr bool operator!=(const B&, const B&) noexcept;

  friend constexpr std::strong_ordering operator<=>(const B&, const B&);
  friend constexpr bool operator<(const B&, const B&);
  friend constexpr bool operator<=(const B&, const B&);
  friend constexpr bool operator>(const B&, const B&);
  friend constexpr bool operator>=(const B&, const B&);
};

struct C {
  friend bool operator==(const C&, const C&); // expected-note {{previous}} extension-note 2{{non-constexpr comparison function declared here}}
  friend bool operator!=(const C&, const C&) = default; // expected-note {{previous}}

  friend std::strong_ordering operator<=>(const C&, const C&); // expected-note {{previous}} extension-note 2{{non-constexpr comparison function declared here}}
  friend bool operator<(const C&, const C&) = default; // expected-note {{previous}}
  friend bool operator<=(const C&, const C&) = default; // expected-note {{previous}}
  friend bool operator>(const C&, const C&) = default; // expected-note {{previous}}
  friend bool operator>=(const C&, const C&) = default; // expected-note {{previous}}
};
struct TestC {
  friend constexpr bool operator==(const C&, const C&); // expected-error {{non-constexpr}}
  friend constexpr bool operator!=(const C&, const C&); // expected-error {{non-constexpr}}

  friend constexpr std::strong_ordering operator<=>(const C&, const C&); // expected-error {{non-constexpr}}
  friend constexpr bool operator<(const C&, const C&); // expected-error {{non-constexpr}}
  friend constexpr bool operator<=(const C&, const C&); // expected-error {{non-constexpr}}
  friend constexpr bool operator>(const C&, const C&); // expected-error {{non-constexpr}}
  friend constexpr bool operator>=(const C&, const C&); // expected-error {{non-constexpr}}
};

struct D {
  A a;
  C c;
  A b;
  friend bool operator==(const D&, const D&) = default; // expected-note {{previous}}
  friend bool operator!=(const D&, const D&) = default; // expected-note {{previous}}

  friend std::strong_ordering operator<=>(const D&, const D&) = default; // expected-note {{previous}}
  friend bool operator<(const D&, const D&) = default; // expected-note {{previous}}
  friend bool operator<=(const D&, const D&) = default; // expected-note {{previous}}
  friend bool operator>(const D&, const D&) = default; // expected-note {{previous}}
  friend bool operator>=(const D&, const D&) = default; // expected-note {{previous}}
};
struct TestD {
  friend constexpr bool operator==(const D&, const D&); // expected-error {{non-constexpr}}
  friend constexpr bool operator!=(const D&, const D&); // expected-error {{non-constexpr}}

  friend constexpr std::strong_ordering operator<=>(const D&, const D&); // expected-error {{non-constexpr}}
  friend constexpr bool operator<(const D&, const D&); // expected-error {{non-constexpr}}
  friend constexpr bool operator<=(const D&, const D&); // expected-error {{non-constexpr}}
  friend constexpr bool operator>(const D&, const D&); // expected-error {{non-constexpr}}
  friend constexpr bool operator>=(const D&, const D&); // expected-error {{non-constexpr}}
};


struct E {
  A a;
  C c; // extension-note 2{{non-constexpr comparison function would be used to compare member 'c'}}
  A b;
  friend constexpr bool operator==(const E&, const E&) = default; // extension-warning {{declared constexpr but invokes a non-constexpr comparison function is a C++23 extension}}
  friend constexpr bool operator!=(const E&, const E&) = default;

  friend constexpr std::strong_ordering operator<=>(const E&, const E&) = default; // extension-warning {{declared constexpr but invokes a non-constexpr comparison function is a C++23 extension}}
  friend constexpr bool operator<(const E&, const E&) = default;
  friend constexpr bool operator<=(const E&, const E&) = default;
  friend constexpr bool operator>(const E&, const E&) = default;
  friend constexpr bool operator>=(const E&, const E&) = default;
};

struct E2 : A, C { // extension-note 2{{non-constexpr comparison function would be used to compare base class 'C'}}
  friend constexpr bool operator==(const E2&, const E2&) = default; // extension-warning {{declared constexpr but invokes a non-constexpr comparison function is a C++23 extension}}
  friend constexpr bool operator!=(const E2&, const E2&) = default;

  friend constexpr std::strong_ordering operator<=>(const E2&, const E2&) = default; // extension-warning {{declared constexpr but invokes a non-constexpr comparison function is a C++23 extension}}
  friend constexpr bool operator<(const E2&, const E2&) = default;
  friend constexpr bool operator<=(const E2&, const E2&) = default;
  friend constexpr bool operator>(const E2&, const E2&) = default;
  friend constexpr bool operator>=(const E2&, const E2&) = default;
};

struct F {
  friend bool operator==(const F&, const F&); // extension-note {{non-constexpr comparison function declared here}}
  friend constexpr bool operator!=(const F&, const F&) = default; // extension-warning {{declared constexpr but invokes a non-constexpr comparison function is a C++23 extension}}

  friend std::strong_ordering operator<=>(const F&, const F&); // extension-note 4{{non-constexpr comparison function declared here}}
  friend constexpr bool operator<(const F&, const F&) = default; // extension-warning {{declared constexpr but invokes a non-constexpr comparison function is a C++23 extension}}
  friend constexpr bool operator<=(const F&, const F&) = default; // extension-warning {{declared constexpr but invokes a non-constexpr comparison function is a C++23 extension}}
  friend constexpr bool operator>(const F&, const F&) = default; // extension-warning {{declared constexpr but invokes a non-constexpr comparison function is a C++23 extension}}
  friend constexpr bool operator>=(const F&, const F&) = default; // extension-warning {{declared constexpr but invokes a non-constexpr comparison function is a C++23 extension}}
};

// No implicit 'constexpr' if it's not the first declaration.
// FIXME: This rule creates problems for reordering of declarations; is this
// really the right model?
struct G;
bool operator==(const G&, const G&); // expected-note {{previous declaration}}
bool operator!=(const G&, const G&); // expected-note {{previous declaration}}
std::strong_ordering operator<=>(const G&, const G&); // expected-note {{previous declaration}}
bool operator<(const G&, const G&); // expected-note {{previous declaration}}
bool operator<=(const G&, const G&); // expected-note {{previous declaration}}
bool operator>(const G&, const G&); // expected-note {{previous declaration}}
bool operator>=(const G&, const G&); // expected-note {{previous declaration}}
struct G {
  friend bool operator==(const G&, const G&) = default; // expected-error {{because it was already declared outside}}
  friend bool operator!=(const G&, const G&) = default; // expected-error {{because it was already declared outside}}

  friend std::strong_ordering operator<=>(const G&, const G&) = default; // expected-error {{because it was already declared outside}}
  friend bool operator<(const G&, const G&) = default; // expected-error {{because it was already declared outside}}
  friend bool operator<=(const G&, const G&) = default; // expected-error {{because it was already declared outside}}
  friend bool operator>(const G&, const G&) = default; // expected-error {{because it was already declared outside}}
  friend bool operator>=(const G&, const G&) = default; // expected-error {{because it was already declared outside}}
};
bool operator==(const G&, const G&);
bool operator!=(const G&, const G&);

std::strong_ordering operator<=>(const G&, const G&);
bool operator<(const G&, const G&);
bool operator<=(const G&, const G&);
bool operator>(const G&, const G&);
bool operator>=(const G&, const G&);

namespace PR44721 {
  template <typename T> bool operator==(T const &, T const &) { return true; }
  template <typename T, typename U> bool operator!=(T const &, U const &) { return true; }
  template <typename T> int operator<=>(T const &, T const &) { return 0; }

  struct S {
    friend bool operator==(const S &, const S &) = default;
    friend bool operator<=>(const S &, const S &) = default;
    int x;
  };
}
