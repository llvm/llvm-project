// RUN: %clang_cc1 -std=c++03 -verify=expected,cxx03,pre14,pre20,pre23,pre2d %s
// RUN: %clang_cc1 -std=c++11 -verify=expected,pre14,pre20,pre23,pre2d %s
// RUN: %clang_cc1 -std=c++14 -verify=expected,pre20,pre23,pre2d %s
// RUN: %clang_cc1 -std=c++17 -verify=expected,pre20,pre23,pre2d %s
// RUN: %clang_cc1 -std=c++20 -verify=expected,cxx20,pre23,pre2d %s
// RUN: %clang_cc1 -std=c++23 -verify=expected,cxx20,pre2d %s
// RUN: %clang_cc1 -std=c++26 -verify=expected,cxx20,pre2d %s
// RUN: %clang_cc1 -std=c++2d -verify=expected,cxx2d %s
// RUN: %clang_cc1 -std=c++2d -Wpre-c++2d-compat -verify=expected,cxx2d,compat %s

// Defaulted postfix increment and decrement operators (P3668R4) are a C++2d
// feature that is available in all language modes as an extension.

struct S {
  int v;
  S &operator++();
  S &operator--();
  S operator++(int) = default;
  // pre2d-warning@-1 {{defaulted postfix increment and decrement operators are a C++2d extension}}
  // compat-warning@-2 {{defaulted postfix increment and decrement operators are incompatible with C++ standards before C++2d}}
  // cxx03-warning@-3 {{defaulted function definitions are a C++11 extension}}
  S operator--(int) = default;
  // pre2d-warning@-1 {{defaulted postfix increment and decrement operators are a C++2d extension}}
  // compat-warning@-2 {{defaulted postfix increment and decrement operators are incompatible with C++ standards before C++2d}}
  // cxx03-warning@-3 {{defaulted function definitions are a C++11 extension}}
};
void use_s(S s) {
  s++;
  s--;
}

struct N {
  N &operator++();
};
N operator++(N &, int) = default;
// pre2d-warning@-1 {{defaulted postfix increment and decrement operators are a C++2d extension}}
// compat-warning@-2 {{defaulted postfix increment and decrement operators are incompatible with C++ standards before C++2d}}
// cxx03-warning@-3 {{defaulted function definitions are a C++11 extension}}
void use_n(N n) { n++; }

// The warning is produced once for a class template, not again for each
// instantiation.
template <typename T> struct W {
  T v;
  W &operator++();
  W operator++(int) = default;
  // pre2d-warning@-1 {{defaulted postfix increment and decrement operators are a C++2d extension}}
  // compat-warning@-2 {{defaulted postfix increment and decrement operators are incompatible with C++ standards before C++2d}}
  // cxx03-warning@-3 {{defaulted function definitions are a C++11 extension}}
};
void use_w(W<int> w, W<long> x) {
  w++;
  x++;
}

// Other functions still cannot be defaulted.
struct Other {
  void f() = default;
  // pre20-error@-1 {{only special member functions may be defaulted}}
  // cxx20-error@-2 {{only special member functions and comparison operators may be defaulted}}
  // cxx2d-error@-3 {{only special member functions, comparison operators, and postfix increment and decrement operators may be defaulted}}
  // cxx03-warning@-4 {{defaulted function definitions are a C++11 extension}}
  Other &operator++() = default;
  // expected-error@-1 {{only the postfix form of 'operator++' can be defaulted}}
  // cxx03-warning@-2 {{defaulted function definitions are a C++11 extension}}
};

// An implicitly deleted defaulted postfix operator.
struct Deleted {
  Deleted operator++(int) = default; // #Deleted
  // expected-warning@#Deleted {{explicitly defaulted postfix increment operator is implicitly deleted}}
  // expected-note@#Deleted 2 {{defaulted 'operator++' is implicitly deleted because there is no viable prefix 'operator++' for an lvalue of type 'Deleted'}}
  // expected-note@#Deleted {{replace 'default' with 'delete'}}
  // expected-note@#Deleted {{explicitly defaulted function was implicitly deleted here}}
  // pre2d-warning@#Deleted {{defaulted postfix increment and decrement operators are a C++2d extension}}
  // compat-warning@#Deleted {{defaulted postfix increment and decrement operators are incompatible with C++ standards before C++2d}}
  // cxx03-warning@#Deleted {{defaulted function definitions are a C++11 extension}}
};
void use_deleted(Deleted d) {
  d++; // expected-error {{object of type 'Deleted' cannot be incremented because its defaulted postfix increment operator is implicitly deleted}}
}

#if __cplusplus >= 201402L
// Unlike other defaulted functions before C++23, a defaulted postfix operator
// may be declared constexpr even if it invokes a non-constexpr function.
struct NotConstexprCompatible {
  NotConstexprCompatible &operator++(); // expected-note {{declared here}}
  constexpr NotConstexprCompatible operator++(int) = default;
  // expected-note@-1 {{non-constexpr function 'operator++' cannot be used in a constant expression}}
  // pre2d-warning@-2 {{defaulted postfix increment and decrement operators are a C++2d extension}}
  // compat-warning@-3 {{defaulted postfix increment and decrement operators are incompatible with C++ standards before C++2d}}
};

struct NotConstexprNonMember {
  NotConstexprNonMember &operator++();
};
constexpr NotConstexprNonMember operator++(NotConstexprNonMember &, int) = default;
// pre2d-warning@-1 {{defaulted postfix increment and decrement operators are a C++2d extension}}
// compat-warning@-2 {{defaulted postfix increment and decrement operators are incompatible with C++ standards before C++2d}}

// Such a function is not usable in constant expressions.
constexpr bool use_not_constexpr_compatible() {
  NotConstexprCompatible n{};
  n++; // expected-note {{in call to 'n.operator++(0)'}}
  return true;
}
static_assert(use_not_constexpr_compatible(), "");
// expected-error@-1 {{static assertion expression is not an integral constant expression}}
// expected-note@-2 {{in call to 'use_not_constexpr_compatible()'}}

// A defaulted postfix operator is implicitly constexpr when it can be.
struct Constexpr {
  int v;
  constexpr Constexpr &operator++() { ++v; return *this; }
  Constexpr operator++(int) = default;
  // pre2d-warning@-1 {{defaulted postfix increment and decrement operators are a C++2d extension}}
  // compat-warning@-2 {{defaulted postfix increment and decrement operators are incompatible with C++ standards before C++2d}}
};
constexpr int use_constexpr() {
  Constexpr c{1};
  Constexpr a = c++;
  return a.v * 10 + c.v;
}
static_assert(use_constexpr() == 12, "");
#endif

#if __cplusplus >= 202002L
struct NotConstevalCompatible {
  NotConstevalCompatible &operator++();
  consteval NotConstevalCompatible operator++(int) = default;
  // pre2d-warning@-1 {{defaulted postfix increment and decrement operators are a C++2d extension}}
  // compat-warning@-2 {{defaulted postfix increment and decrement operators are incompatible with C++ standards before C++2d}}
};
struct Consteval {
  int v;
  constexpr Consteval &operator++() { ++v; return *this; }
  consteval Consteval operator++(int) = default;
  // pre2d-warning@-1 {{defaulted postfix increment and decrement operators are a C++2d extension}}
  // compat-warning@-2 {{defaulted postfix increment and decrement operators are incompatible with C++ standards before C++2d}}
};
consteval int use_consteval() {
  Consteval c{1};
  Consteval a = c++;
  return a.v * 10 + c.v;
}
static_assert(use_consteval() == 12);

// Defaulted postfix operators may be constrained in every mode that has
// requires-clauses.
template <typename T> concept Small = sizeof(T) <= 4;
template <typename T> struct Constrained {
  T v;
  constexpr Constrained &operator++() requires Small<T> { ++v; return *this; }
  constexpr Constrained &operator--() { --v; return *this; }
  Constrained operator++(int) requires Small<T> = default;
  // pre2d-warning@-1 {{defaulted postfix increment and decrement operators are a C++2d extension}}
  // compat-warning@-2 {{defaulted postfix increment and decrement operators are incompatible with C++ standards before C++2d}}
  friend Constrained operator--(Constrained &, int) requires Small<T> = default;
  // pre2d-warning@-1 {{defaulted postfix increment and decrement operators are a C++2d extension}}
  // compat-warning@-2 {{defaulted postfix increment and decrement operators are incompatible with C++ standards before C++2d}}
};
constexpr int use_constrained() {
  Constrained<int> c{1};
  Constrained<int> a = c++;
  Constrained<int> b = c--;
  return a.v * 100 + b.v * 10 + c.v;
}
static_assert(use_constrained() == 121);
template <typename T> concept PostIncrementable = requires(T t) { t++; };
static_assert(PostIncrementable<Constrained<int>>);
static_assert(!PostIncrementable<Constrained<long long>>);
void use_constrained_unsatisfied(Constrained<long long> c) {
  c++; // expected-error {{cannot increment value of type 'Constrained<long long>'}}
  c--; // expected-error {{cannot decrement value of type 'Constrained<long long>'}}
}

// An unsatisfied constraint on the prefix operator deletes the defaulted
// postfix operator.
template <typename T> struct ConstrainedPrefix {
  ConstrainedPrefix &operator++() requires Small<T>;
  ConstrainedPrefix operator++(int) = default; // #ConstrainedPrefix
  // pre2d-warning@#ConstrainedPrefix {{defaulted postfix increment and decrement operators are a C++2d extension}}
  // compat-warning@#ConstrainedPrefix {{defaulted postfix increment and decrement operators are incompatible with C++ standards before C++2d}}
  // expected-note@#ConstrainedPrefix {{explicitly defaulted function was implicitly deleted here}}
  // expected-note@#ConstrainedPrefix {{defaulted 'operator++' is implicitly deleted because there is no viable prefix 'operator++' for an lvalue of type 'ConstrainedPrefix<long long>'}}
};
void use_constrained_prefix(ConstrainedPrefix<int> a, ConstrainedPrefix<long long> b) {
  a++;
  b++; // expected-error {{object of type 'ConstrainedPrefix<long long>' cannot be incremented because its defaulted postfix increment operator is implicitly deleted}}
}
#endif
