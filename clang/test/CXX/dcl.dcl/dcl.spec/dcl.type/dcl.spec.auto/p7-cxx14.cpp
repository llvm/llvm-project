// RUN: %clang_cc1 -verify -std=c++2b -verify=expected,cxx2b    %s
// RUN: %clang_cc1 -verify -std=c++20 -verify=expected,cxx14_20 %s
// RUN: %clang_cc1 -verify -std=c++14 -verify=expected,cxx14_20 %s

namespace std {
  template<typename T> struct initializer_list {
    const T *p;
    unsigned long n;
    initializer_list(const T *p, unsigned long n);
  };
}

int i;
int &&f();

template <typename T>
void overloaded_fn(T); // expected-note {{possible target}}

using Int = int;
using IntLRef = int&;
using IntRRef = int&&;
using InitListInt = std::initializer_list<int>;
using IntPtr = int*;

auto x3a = i;
decltype(auto) x3d = i;
using Int = decltype(x3a);
using Int = decltype(x3d);

auto x4a = (i);
decltype(auto) x4d = (i);
using Int = decltype(x4a);
using IntLRef = decltype(x4d); // cxx2b-note {{previous definition is here}}

auto x5a = f();
decltype(auto) x5d = f();
using Int = decltype(x5a);
using IntRRef = decltype(x5d);

auto x6a = { 1, 2 };
decltype(auto) x6d = { 1, 2 }; // expected-error {{cannot deduce 'decltype(auto)' from initializer list}}
using InitListInt = decltype(x6a);

auto *x7a = &i;
decltype(auto) *x7d = &i; // expected-error {{cannot form pointer to 'decltype(auto)'}}
using IntPtr = decltype(x7a);

struct S {};

decltype(auto) f1();
decltype(auto) (*f2)(); // expected-error {{'decltype(auto)' can only be used as a return type in a function declaration}} expected-error {{requires an initializer}}
decltype(auto) *f3(); // expected-error {{cannot form pointer to 'decltype(auto)'}}
const decltype(auto) f4(); // expected-error {{'decltype(auto)' cannot be combined with other type specifiers}}
typedef decltype(auto) f5(); // expected-error {{'decltype(auto)' not allowed in typedef}}
decltype(auto) ((((((f6))))())); // ok
decltype(auto) f7()(); // expected-error {{'decltype(auto)' can only be used as a return type in a function declaration}}
decltype(auto) (S::*f8)(); // expected-error {{'decltype(auto)' can only be used as a return type in a function declaration}} expected-error {{requires an initializer}}
decltype(auto) &f9(); // expected-error {{cannot form reference to 'decltype(auto)'}}
decltype(auto) (&f10())[10]; // expected-error {{cannot form array of 'decltype(auto)'}}

decltype(auto) ((((((v1)))))) = 0; // ok
decltype(auto) v2[1] = { 0 }; // expected-error {{cannot form array of 'decltype(auto)'}}
decltype(auto) &v3 = { 0 }; // expected-error {{cannot form reference to 'decltype(auto)'}}
decltype(auto) *v4 = { 0 }; // expected-error {{cannot form pointer to 'decltype(auto)'}}
decltype(auto) v5 = &overloaded_fn; // expected-error {{could not be resolved}}

auto multi1a = 0, &multi1b = multi1a;
auto multi1c = multi1a, multi1d = multi1b;
decltype(auto) multi1e = multi1a, multi1f = multi1b; // expected-error {{'decltype(auto)' deduced as 'int' in declaration of 'multi1e' and deduced as 'int &' in declaration of 'multi1f'}}

auto f1a() { return 0; }
decltype(auto) f1d() { return 0; }
using Int = decltype(f1a());
using Int = decltype(f1d());

auto f2a(int n) { return n; }
decltype(auto) f2d(int n) { return n; }
using Int = decltype(f2a(0));
using Int = decltype(f2d(0));

auto f3a(int n) { return (n); }
decltype(auto) f3d(int n) { return (n); } // expected-warning {{reference to stack memory}}
using Int = decltype(f3a(0));
using IntLRef = decltype(f3d(0)); // cxx2b-error {{type alias redefinition with different types ('decltype(f3d(0))' (aka 'int &&') vs 'decltype(x4d)' (aka 'int &'))}}

auto f4a(int n) { return f(); }
decltype(auto) f4d(int n) { return f(); }
using Int = decltype(f4a(0));
using IntRRef = decltype(f4d(0));

auto f5aa(int n) { auto x = f(); return x; }
auto f5ad(int n) { decltype(auto) x = f(); return x; }
decltype(auto) f5da(int n) { auto x = f(); return x; }
decltype(auto) f5dd(int n) { decltype(auto) x = f(); return x; } // cxx14_20-error {{rvalue reference to type 'int' cannot bind to lvalue}}
using Int = decltype(f5aa(0));
using Int = decltype(f5ad(0));
using Int = decltype(f5da(0));

auto init_list_1() { return { 1, 2, 3 }; } // expected-error {{cannot deduce return type from initializer list}}
decltype(auto) init_list_2() { return { 1, 2, 3 }; } // expected-error {{cannot deduce return type from initializer list}}
