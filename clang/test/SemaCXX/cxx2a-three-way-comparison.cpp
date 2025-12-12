// RUN: %clang_cc1 -std=c++2a -verify %s -Wzero-as-null-pointer-constant

// Keep this test before any declarations of operator<=>.
namespace PR44786 {
  template<typename T> void f(decltype(T{} <=> T{})) {} // expected-note {{previous}}

  struct S {};
  int operator<=>(S const &, S const &);
  template<typename T> void f(decltype(T{} <=> T{})) {} // expected-error {{redefinition}}
}

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

namespace PR47893 {
  struct A {
    void operator<=>(const A&) const;
  };
  template<typename T> auto f(T a, T b) -> decltype(a < b) = delete;
  int &f(...);
  int &r = f(A(), A());
}

namespace PR44325 {
  struct cmp_cat {};
  bool operator<(cmp_cat, void*);
  bool operator>(cmp_cat, int cmp_cat::*);

  struct X {};
  cmp_cat operator<=>(X, X);

  bool b1 = X() < X(); // no warning
  bool b2 = X() > X(); // no warning

  // FIXME: It's not clear whether warning here is useful, but we can't really
  // tell that this is a comparison category in general. This is probably OK,
  // as comparisons against zero are only really intended for use in the
  // implicit rewrite rules, not for explicit use by programs.
  bool c = cmp_cat() < 0; // expected-warning {{zero as null pointer constant}}
}

namespace GH137452 {
struct comparable_t {
    __attribute__((vector_size(32))) double numbers;           // expected-note {{declared here}}
    auto operator<=>(const comparable_t& rhs) const = default; // expected-warning {{explicitly defaulted three-way comparison operator is implicitly deleted}} \
                                                                  expected-note {{replace 'default' with 'delete'}} \
                                                                  expected-note {{defaulted 'operator<=>' is implicitly deleted because defaulted comparison of vector types is not supported}}
};
} // namespace GH137452

namespace GH170015 {
// This test ensures that the compiler enforces strict type checking on the 
// static members of comparison category types.
// Previously, a mismatch (e.g., equivalent being an int) could crash the compiler.
}

namespace std {
  struct partial_ordering {
    // Malformed: 'equivalent' should be of type 'partial_ordering', not 'int'.
    static constexpr int equivalent = 0; 
    static constexpr int less = -1;
    static constexpr int greater = 1;
    static constexpr int unordered = 2;
  };
}

namespace GH170015 {
  void f() {
    float a = 0.0f, b = 0.0f;
    // We expect the compiler to complain that the type form is wrong 
    // (because the static members are ints, not objects).
    auto res = a <=> b; // expected-error {{standard library implementation of 'std::partial_ordering' is not supported; the type does not have the expected form}}
  }
}