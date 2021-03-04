// RUN: %clang_cc1 -std=c++2a -verify %s

struct A1 {
  int x;
  int &y; // expected-note 9{{because class 'A1' has a reference member}}

  bool operator==(const A1&) const = default; // expected-warning {{implicitly deleted}} expected-note 2{{deleted here}}
  bool operator<=>(const A1&) const = default; // expected-warning {{implicitly deleted}} expected-note 5{{deleted here}}
};
struct A2 {
  int x;
  int &y;

  bool operator==(const A2&) const;
  bool operator!=(const A2&) const = default;

  int operator<=>(const A2&) const;
  bool operator<(const A2&) const = default;
  bool operator<=(const A2&) const = default;
  bool operator>(const A2&) const = default;
  bool operator>=(const A2&) const = default;
};
void f(A1 a) {
  void(a == a); // expected-error {{deleted}}
  void(a != a); // expected-error {{deleted}}
  void(a <=> a); // expected-error {{deleted}}
  void(a < a); // expected-error {{deleted}}
  void(a <= a); // expected-error {{deleted}}
  void(a > a); // expected-error {{deleted}}
  void(a >= a); // expected-error {{deleted}}
}
void f(A2 a) {
  void(a == a);
  void(a != a);
  void(a <=> a);
  void(a < a);
  void(a <= a);
  void(a > a);
  void(a >= a);
}

struct A3 {
  int &x; // expected-note {{because class 'A3' has a reference member}}

  bool operator==(const A3 &) const = default; // expected-warning {{implicitly deleted}}
  bool operator<(const A3 &) const = default;  // expected-warning {{implicitly deleted}}
  // expected-note@-1 {{because there is no viable comparison function}}
};

struct B1 {
  struct {
    int x;
    int &y; // expected-note 2{{because class 'B1' has a reference member}}
  };

  bool operator==(const B1&) const = default; // expected-warning {{implicitly deleted}}
  bool operator<=>(const B1&) const = default; // expected-warning {{implicitly deleted}}
};

struct B2 {
  struct {
    int x;
    int &y;
  };

  bool operator==(const B2&) const;
  bool operator!=(const B2&) const = default;

  bool operator<=>(const B2&) const;
  bool operator<(const B2&) const = default;
  bool operator<=(const B2&) const = default;
  bool operator>(const B2&) const = default;
  bool operator>=(const B2&) const = default;
};

union C1 {
  int a;

  bool operator==(const C1&) const = default; // expected-warning {{implicitly deleted}} expected-note {{because 'C1' is a union }}
  bool operator<=>(const C1&) const = default; // expected-warning {{implicitly deleted}} expected-note {{because 'C1' is a union }}
};

union C2 {
  int a;

  bool operator==(const C2&) const;
  bool operator!=(const C2&) const = default;

  bool operator<=>(const C2&) const;
  bool operator<(const C2&) const = default;
  bool operator<=(const C2&) const = default;
  bool operator>(const C2&) const = default;
  bool operator>=(const C2&) const = default;
};

struct D1 {
  union {
    int a;
  };

  bool operator==(const D1&) const = default; // expected-warning {{implicitly deleted}} expected-note {{because 'D1' is a union-like class}}
  bool operator<=>(const D1&) const = default; // expected-warning {{implicitly deleted}} expected-note {{because 'D1' is a union-like class}}
};
struct D2 {
  union {
    int a;
  };

  bool operator==(const D2&) const;
  bool operator!=(const D2&) const = default;

  bool operator<=>(const D2&) const;
  bool operator<(const D2&) const = default;
  bool operator<=(const D2&) const = default;
  bool operator>(const D2&) const = default;
  bool operator>=(const D2&) const = default;
};

union E1 {
  bool operator==(const E1&) const = default;
  bool operator!=(const E1&) const = default;

  bool operator<=>(const E1&) const = default;
  bool operator<(const E1&) const = default;
  bool operator<=(const E1&) const = default;
  bool operator>(const E1&) const = default;
  bool operator>=(const E1&) const = default;
};
union E2 {
  bool operator==(const E2&) const = default;
  bool operator!=(const E2&) const = default;

  bool operator<=>(const E2&) const = default;
  bool operator<(const E2&) const = default;
  bool operator<=(const E2&) const = default;
  bool operator>(const E2&) const = default;
  bool operator>=(const E2&) const = default;
};

struct F;
bool operator==(const F&, const F&);
bool operator!=(const F&, const F&);
bool operator<=>(const F&, const F&);
bool operator<(const F&, const F&);
struct F {
  union { int a; };
  friend bool operator==(const F&, const F&) = default; // expected-error {{defaulting this equality comparison operator would delete it after its first declaration}} expected-note {{implicitly deleted because 'F' is a union-like class}}
  friend bool operator!=(const F&, const F&) = default;
  friend bool operator<=>(const F&, const F&) = default; // expected-error {{defaulting this three-way comparison operator would delete it after its first declaration}} expected-note {{implicitly deleted because 'F' is a union-like class}}
  friend bool operator<(const F&, const F&) = default;
};
