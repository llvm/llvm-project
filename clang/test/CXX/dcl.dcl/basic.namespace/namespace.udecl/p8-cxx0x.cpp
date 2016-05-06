// RUN: %clang_cc1 -fsyntax-only -std=c++98 -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s
// RUN: not %clang_cc1 -fsyntax-only -std=c++98 -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck --check-prefix=CXX98 %s
// RUN: not %clang_cc1 -fsyntax-only -std=c++11 -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck --check-prefix=CXX11 %s
// C++0x N2914.

struct X {
  int i;
  static int a;
  enum E { e };
};

using X::i; // expected-error{{using declaration cannot refer to class member}}
using X::s; // expected-error{{using declaration cannot refer to class member}}
using X::e; // expected-error{{using declaration cannot refer to class member}}
using X::E::e; // expected-error{{using declaration cannot refer to class member}} expected-warning 0-1{{C++11}}
#if __cplusplus < 201103L
// expected-note@-3 {{use a const variable}}
// expected-note@-3 {{use a const variable}}
// CXX98-NOT: fix-it:"{{.*}}":{[[@LINE-5]]:
// CXX98-NOT: fix-it:"{{.*}}":{[[@LINE-5]]:
#else
// expected-note@-8 {{use a constexpr variable}}
// expected-note@-8 {{use a constexpr variable}}
// CXX11: fix-it:"{{.*}}":{[[@LINE-10]]:1-[[@LINE-10]]:6}:"constexpr auto e = "
// CXX11: fix-it:"{{.*}}":{[[@LINE-10]]:1-[[@LINE-10]]:6}:"constexpr auto e = "
#endif

void f() {
  using X::i; // expected-error{{using declaration cannot refer to class member}}
  using X::s; // expected-error{{using declaration cannot refer to class member}}
  using X::e; // expected-error{{using declaration cannot refer to class member}}
  using X::E::e; // expected-error{{using declaration cannot refer to class member}} expected-warning 0-1{{C++11}}
#if __cplusplus < 201103L
  // expected-note@-3 {{use a const variable}}
  // expected-note@-3 {{use a const variable}}
  // CXX98-NOT: fix-it:"{{.*}}":{[[@LINE-5]]:
  // CXX98-NOT: fix-it:"{{.*}}":{[[@LINE-5]]:
#else
  // expected-note@-8 {{use a constexpr variable}}
  // expected-note@-8 {{use a constexpr variable}}
  // CXX11: fix-it:"{{.*}}":{[[@LINE-10]]:3-[[@LINE-10]]:8}:"constexpr auto e = "
  // CXX11: fix-it:"{{.*}}":{[[@LINE-10]]:3-[[@LINE-10]]:8}:"constexpr auto e = "
#endif
}

template <typename T>
struct PR21933 : T {
  static void StaticFun() { using T::member; } // expected-error{{using declaration cannot refer to class member}}
};

struct S {
  static int n;
  struct Q {};
  enum E {};
  typedef Q T;
  void f();
  static void g();
};

using S::n; // expected-error{{class member}} expected-note {{use a reference instead}}
#if __cplusplus < 201103L
// CXX98-NOT: fix-it:"{{.*}}":{[[@LINE-2]]
#else
// CXX11: fix-it:"{{.*}}":{[[@LINE-4]]:1-[[@LINE-4]]:6}:"auto &n = "
#endif

using S::Q; // expected-error{{class member}}
#if __cplusplus < 201103L
// expected-note@-2 {{use a typedef declaration instead}}
// CXX98: fix-it:"{{.*}}":{[[@LINE-3]]:1-[[@LINE-3]]:6}:"typedef"
// CXX98: fix-it:"{{.*}}":{[[@LINE-4]]:11-[[@LINE-4]]:11}:" Q"
#else
// expected-note@-6 {{use an alias declaration instead}}
// CXX11: fix-it:"{{.*}}":{[[@LINE-7]]:7-[[@LINE-7]]:7}:"Q = "
#endif

using S::E; // expected-error{{class member}}
#if __cplusplus < 201103L
// expected-note@-2 {{use a typedef declaration instead}}
// CXX98: fix-it:"{{.*}}":{[[@LINE-3]]:1-[[@LINE-3]]:6}:"typedef"
// CXX98: fix-it:"{{.*}}":{[[@LINE-4]]:11-[[@LINE-4]]:11}:" E"
#else
// expected-note@-6 {{use an alias declaration instead}}
// CXX11: fix-it:"{{.*}}":{[[@LINE-7]]:7-[[@LINE-7]]:7}:"E = "
#endif

using S::T; // expected-error{{class member}}
#if __cplusplus < 201103L
// expected-note@-2 {{use a typedef declaration instead}}
// CXX98: fix-it:"{{.*}}":{[[@LINE-3]]:1-[[@LINE-3]]:6}:"typedef"
// CXX98: fix-it:"{{.*}}":{[[@LINE-4]]:11-[[@LINE-4]]:11}:" T"
#else
// expected-note@-6 {{use an alias declaration instead}}
// CXX11: fix-it:"{{.*}}":{[[@LINE-7]]:7-[[@LINE-7]]:7}:"T = "
#endif

using S::f; // expected-error{{class member}}
using S::g; // expected-error{{class member}}

void h() {
  using S::n; // expected-error{{class member}} expected-note {{use a reference instead}}
#if __cplusplus < 201103L
  // CXX98-NOT: fix-it:"{{.*}}":{[[@LINE-2]]
#else
  // CXX11: fix-it:"{{.*}}":{[[@LINE-4]]:3-[[@LINE-4]]:8}:"auto &n = "
#endif

  using S::Q; // expected-error{{class member}}
#if __cplusplus < 201103L
  // expected-note@-2 {{use a typedef declaration instead}}
  // CXX98: fix-it:"{{.*}}":{[[@LINE-3]]:3-[[@LINE-3]]:8}:"typedef"
  // CXX98: fix-it:"{{.*}}":{[[@LINE-4]]:13-[[@LINE-4]]:13}:" Q"
#else
  // expected-note@-6 {{use an alias declaration instead}}
  // CXX11: fix-it:"{{.*}}":{[[@LINE-7]]:9-[[@LINE-7]]:9}:"Q = "
#endif

  using S::E; // expected-error{{class member}}
#if __cplusplus < 201103L
  // expected-note@-2 {{use a typedef declaration instead}}
  // CXX98: fix-it:"{{.*}}":{[[@LINE-3]]:3-[[@LINE-3]]:8}:"typedef"
  // CXX98: fix-it:"{{.*}}":{[[@LINE-4]]:13-[[@LINE-4]]:13}:" E"
#else
  // expected-note@-6 {{use an alias declaration instead}}
  // CXX11: fix-it:"{{.*}}":{[[@LINE-7]]:9-[[@LINE-7]]:9}:"E = "
#endif

  using S::T; // expected-error{{class member}}
#if __cplusplus < 201103L
  // expected-note@-2 {{use a typedef declaration instead}}
  // CXX98: fix-it:"{{.*}}":{[[@LINE-3]]:3-[[@LINE-3]]:8}:"typedef"
  // CXX98: fix-it:"{{.*}}":{[[@LINE-4]]:13-[[@LINE-4]]:13}:" T"
#else
  // expected-note@-6 {{use an alias declaration instead}}
  // CXX11: fix-it:"{{.*}}":{[[@LINE-7]]:9-[[@LINE-7]]:9}:"T = "
#endif

  using S::f; // expected-error{{class member}}
  using S::g; // expected-error{{class member}}
}
