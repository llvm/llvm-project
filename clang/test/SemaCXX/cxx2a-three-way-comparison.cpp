// RUN: %clang_cc1 -std=c++2a -verify %s

struct A {};
constexpr int operator<=>(A a, A b) { return 42; }
static_assert(operator<=>(A(), A()) == 42);

int operator<=>(); // expected-error {{overloaded 'operator<=>' must have at least one parameter of class or enumeration type}}
int operator<=>(A); // expected-error {{overloaded 'operator<=>' must be a binary operator}}
int operator<=>(int, int); // expected-error {{overloaded 'operator<=>' must have at least one parameter of class or enumeration type}}
int operator<=>(A, A, A); // expected-error {{overloaded 'operator<=>' must be a binary operator}}
int operator<=>(A, A, ...); // expected-error {{overloaded 'operator<=>' cannot be variadic}}
int operator<=>(int, A = {}); // expected-error {{parameter of overloaded 'operator<=>' cannot have a default argument}}

struct B {
  int &operator<=>(int);
  friend int operator<=>(A, B);

  friend int operator<=>(int, int); // expected-error {{overloaded 'operator<=>' must have at least one parameter of class or enumeration type}}
  void operator<=>(); // expected-error {{overloaded 'operator<=>' must be a binary operator}};
  void operator<=>(A, ...); // expected-error {{overloaded 'operator<=>' cannot be variadic}}
  void operator<=>(A, A); // expected-error {{overloaded 'operator<=>' must be a binary operator}};
};

int &r = B().operator<=>(0);
