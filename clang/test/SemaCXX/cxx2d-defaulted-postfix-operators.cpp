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
// Before C++23, an explicitly-defaulted function may only be declared
// constexpr if it is constexpr-compatible.
struct NotConstexprCompatible {
  NotConstexprCompatible &operator++(); // pre23-note {{non-constexpr prefix 'operator++' declared here}}
  constexpr NotConstexprCompatible operator++(int) = default;
  // pre23-error@-1 {{defaulted definition of postfix increment operator cannot be declared constexpr because it invokes a non-constexpr function}}
  // pre2d-warning@-2 {{defaulted postfix increment and decrement operators are a C++2d extension}}
  // compat-warning@-3 {{defaulted postfix increment and decrement operators are incompatible with C++ standards before C++2d}}
};

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
