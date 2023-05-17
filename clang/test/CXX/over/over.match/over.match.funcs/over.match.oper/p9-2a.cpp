// RUN: %clang_cc1 -std=c++2a -verify %s

// ... return type shall be cv bool ...
namespace not_bool {
  struct X {} x;
  struct Y {} y;
  double operator==(X, Y); // expected-note 4{{here}}
  bool a = x == y; // ok
  bool b = y == x; // expected-error {{return type 'double' of selected 'operator==' function for rewritten '==' comparison is not 'bool'}}
  bool c = x != y; // expected-error {{return type 'double' of selected 'operator==' function for rewritten '!=' comparison is not 'bool'}}
  bool d = y != x; // expected-error {{return type 'double' of selected 'operator==' function for rewritten '!=' comparison is not 'bool'}}

  // cv-qualifiers are OK
  const bool operator==(Y, X);
  bool e = y != x; // ok

  // We don't prefer a function with bool return type over one witn non-bool return type.
  bool f = x != y; // expected-error {{return type 'double' of selected 'operator==' function for rewritten '!=' comparison is not 'bool'}}

  // As an extension, we permit integral and unscoped enumeration types too.
  // These are used by popular C++ libraries such as ICU.
  struct Z {} z;
  int operator==(X, Z); // expected-note {{here}}
  bool g = z == x; // expected-warning {{ISO C++20 requires return type of selected 'operator==' function for rewritten '==' comparison to be 'bool', not 'int'}}

  enum E {};
  E operator==(Y, Z); // expected-note {{here}}
  bool h = z == y; // expected-warning {{ISO C++20 requires return type of selected 'operator==' function for rewritten '==' comparison to be 'bool', not 'E'}}
}

struct X { bool equal; };
struct Y {};
constexpr bool operator==(X x, Y) { return x.equal; }

static_assert(X{true} == Y{});
static_assert(X{false} == Y{}); // expected-error {{failed}}

// x == y -> y == x
static_assert(Y{} == X{true});
static_assert(Y{} == X{false}); // expected-error {{failed}}

// x != y -> !(x == y)
static_assert(X{true} != Y{}); // expected-error {{failed}}
static_assert(X{false} != Y{});

// x != y -> !(y == x)
static_assert(Y{} != X{true}); // expected-error {{failed}}
static_assert(Y{} != X{false});
