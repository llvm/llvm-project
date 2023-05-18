// RUN: %clang_cc1 -std=c++20 -verify %s

struct A {
  bool operator!=(const A&) const = default; // expected-warning {{explicitly defaulted equality comparison operator is implicitly deleted}} expected-note{{replace 'default'}}
  // expected-note@-1 {{defaulted 'operator!=' is implicitly deleted because there is no viable 'operator==' for 'A'}}
};

struct A2 {
  struct E {};
  E e;
  bool operator==(const A2&) const = default; // expected-warning {{explicitly defaulted equality comparison operator is implicitly deleted}} expected-note{{replace 'default'}}
  // expected-note@-2 {{defaulted 'operator==' is implicitly deleted because there is no viable 'operator==' for member 'e'}}
};


struct Q {};
bool operator!=(Q, Q); // expected-note {{defaulted 'operator!=' is implicitly deleted because this non-rewritten comparison function would be the best match for the comparison}}
struct B {
  operator Q() const;
  bool operator!=(const B&) const = default; // expected-warning {{explicitly defaulted equality comparison operator is implicitly deleted}} expected-note{{replace 'default'}}
};

struct R {};
bool operator==(R, R);
struct B2 {
  operator R() const;
  bool operator!=(const B2&) const = default; // OK! Converts to use rewritten R comparison.
};

struct C {
  operator int() const; // expected-note {{defaulted 'operator!=' is implicitly deleted because a builtin comparison function using this conversion would be the best match for the comparison}}
  bool operator!=(const C&) const = default; // expected-warning {{explicitly defaulted equality comparison operator is implicitly deleted}} expected-note{{replace 'default'}}
};

struct D {
  bool operator<(const D&) const = default; // expected-warning {{explicitly defaulted relational comparison operator is implicitly deleted}} expected-note{{replace 'default'}}
  // expected-note@-1 {{defaulted 'operator<' is implicitly deleted because there is no viable three-way comparison function for 'D'}}
};

bool operator<(Q, Q); // expected-note {{defaulted 'operator<' is implicitly deleted because this non-rewritten comparison function would be the best match for the comparison}}
struct E {
  operator Q() const;
  bool operator<(const E&) const = default; // expected-warning {{explicitly defaulted relational comparison operator is implicitly deleted}} expected-note{{replace 'default'}}
};

int operator<=>(R, R);
struct E2 {
  operator R() const;
  bool operator<(const E2&) const = default;
};

struct F {
  operator int() const; // expected-note {{defaulted 'operator<' is implicitly deleted because a builtin comparison function using this conversion would be the best match for the comparison}}
  bool operator<(const F&) const = default; // expected-warning {{explicitly defaulted relational comparison operator is implicitly deleted}} expected-note{{replace 'default'}}
};
