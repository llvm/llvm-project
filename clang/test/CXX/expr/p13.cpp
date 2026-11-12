// RUN: %clang_cc1 -std=c++1z -verify %s -fexceptions -fcxx-exceptions -Wno-dynamic-exception-spec

struct X {};
struct Y : X {};

using A = void (*)() noexcept;
using B = void (*)();
using C = void (X::*)() noexcept;
using D = void (X::*)();
using E = void (Y::*)() noexcept;
using F = void (Y::*)();

void f(A a, B b, C c, D d, E e, F f, bool k) {
  a = k ? a : b; // expected-error {{incompatible function pointer types assigning to 'A' (aka 'void (*)() noexcept') from 'void (*)()'}}
  b = k ? a : b;

  c = k ? c : d; // expected-error {{different exception specifications}}
  d = k ? c : d;

  e = k ? c : f; // expected-error {{different exception specifications}}
  e = k ? d : e; // expected-error {{different exception specifications}}
  f = k ? c : f;
  f = k ? d : e;

  const A ak = a;
  const B bk = b;
  const A &ak2 = k ? ak : ak;
  const A &ak3 = k ? ak : bk; // expected-error {{could not bind}}
  const B &bk3 = k ? ak : bk;
}

namespace dynamic_exception_spec {
  // Prior to P0012, we had:
  //   "[...] the target entity shall allow at least the exceptions allowed
  //   by the source value in the assignment or initialization"
  //
  // There's really only one way we can coherently apply this to conditional
  // expressions: this must hold no matter which branch was taken.
  using X = void (*)() throw(int);
  using Y = void (*)() throw(float);
  using Z = void (*)() throw(int, float);
  void g(X x, Y y, Z z, bool k) {
    x = k ? X() : Y(); // expected-warning {{not superset}}
    y = k ? X() : Y(); // expected-warning {{not superset}}
    z = k ? X() : Y();

    x = k ? x : y; // expected-warning {{not superset}}
    y = k ? x : y; // expected-warning {{not superset}}
    z = k ? x : y;
  }
}
