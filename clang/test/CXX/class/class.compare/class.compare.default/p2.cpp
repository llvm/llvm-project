// RUN: %clang_cc1 -std=c++2a -verify %s

struct A {
  int x;
  int &y; // expected-note 7{{reference member 'y' declared here}}

  bool operator==(const A&) const = default; // expected-error {{cannot default 'operator==' in class 'A' with reference member}}
  bool operator!=(const A&) const = default; // expected-warning {{ISO C++2a does not allow defaulting 'operator!=' in class 'A' with reference member}}

  bool operator<=>(const A&) const = default; // expected-error {{cannot default 'operator<=>' in class 'A' with reference member}}
  bool operator<(const A&) const = default; // expected-warning {{ISO C++2a does not allow defaulting 'operator<' in class 'A' with reference member}}
  bool operator<=(const A&) const = default; // expected-warning {{ISO C++2a does not allow defaulting 'operator<=' in class 'A' with reference member}}
  bool operator>(const A&) const = default; // expected-warning {{ISO C++2a does not allow defaulting 'operator>' in class 'A' with reference member}}
  bool operator>=(const A&) const = default; // expected-warning {{ISO C++2a does not allow defaulting 'operator>=' in class 'A' with reference member}}
};

struct B {
  struct {
    int x;
    int &y; // expected-note 7{{reference member 'y' declared here}}
  };

  bool operator==(const B&) const = default; // expected-error {{cannot default 'operator==' in class 'B' with reference member}}
  bool operator!=(const B&) const = default; // expected-warning {{ISO C++2a does not allow defaulting 'operator!=' in class 'B' with reference member}}

  bool operator<=>(const B&) const = default; // expected-error {{cannot default 'operator<=>' in class 'B' with reference member}}
  bool operator<(const B&) const = default; // expected-warning {{ISO C++2a does not allow defaulting 'operator<' in class 'B' with reference member}}
  bool operator<=(const B&) const = default; // expected-warning {{ISO C++2a does not allow defaulting 'operator<=' in class 'B' with reference member}}
  bool operator>(const B&) const = default; // expected-warning {{ISO C++2a does not allow defaulting 'operator>' in class 'B' with reference member}}
  bool operator>=(const B&) const = default; // expected-warning {{ISO C++2a does not allow defaulting 'operator>=' in class 'B' with reference member}}
};

union C {
  int a;

  bool operator==(const C&) const = default; // expected-error {{cannot default 'operator==' in union 'C'}}
  bool operator!=(const C&) const = default; // expected-warning {{ISO C++2a does not allow defaulting 'operator!=' in union 'C'}}

  bool operator<=>(const C&) const = default; // expected-error {{cannot default 'operator<=>' in union 'C'}}
  bool operator<(const C&) const = default; // expected-warning {{ISO C++2a does not allow defaulting 'operator<' in union 'C'}}
  bool operator<=(const C&) const = default; // expected-warning {{ISO C++2a does not allow defaulting 'operator<=' in union 'C'}}
  bool operator>(const C&) const = default; // expected-warning {{ISO C++2a does not allow defaulting 'operator>' in union 'C'}}
  bool operator>=(const C&) const = default; // expected-warning {{ISO C++2a does not allow defaulting 'operator>=' in union 'C'}}
};

struct D {
  union {
    int a;
  };

  bool operator==(const D&) const = default; // expected-error {{cannot default 'operator==' in union-like class 'D'}}
  bool operator!=(const D&) const = default; // expected-warning {{ISO C++2a does not allow defaulting 'operator!=' in union-like class 'D'}}

  bool operator<=>(const D&) const = default; // expected-error {{cannot default 'operator<=>' in union-like class 'D'}}
  bool operator<(const D&) const = default; // expected-warning {{ISO C++2a does not allow defaulting 'operator<' in union-like class 'D'}}
  bool operator<=(const D&) const = default; // expected-warning {{ISO C++2a does not allow defaulting 'operator<=' in union-like class 'D'}}
  bool operator>(const D&) const = default; // expected-warning {{ISO C++2a does not allow defaulting 'operator>' in union-like class 'D'}}
  bool operator>=(const D&) const = default; // expected-warning {{ISO C++2a does not allow defaulting 'operator>=' in union-like class 'D'}}
};

union E {
  bool operator==(const E&) const = default; // expected-warning {{ISO C++2a does not allow defaulting 'operator==' in union 'E' despite it having no variant members}}
  bool operator!=(const E&) const = default; // expected-warning {{ISO C++2a does not allow defaulting 'operator!=' in union 'E'}}

  bool operator<=>(const E&) const = default; // expected-warning {{ISO C++2a does not allow defaulting 'operator<=>' in union 'E' despite it having no variant members}}
  bool operator<(const E&) const = default; // expected-warning {{ISO C++2a does not allow defaulting 'operator<' in union 'E'}}
  bool operator<=(const E&) const = default; // expected-warning {{ISO C++2a does not allow defaulting 'operator<=' in union 'E'}}
  bool operator>(const E&) const = default; // expected-warning {{ISO C++2a does not allow defaulting 'operator>' in union 'E'}}
  bool operator>=(const E&) const = default; // expected-warning {{ISO C++2a does not allow defaulting 'operator>=' in union 'E'}}
};
