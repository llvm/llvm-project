// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -fno-modules-error-recovery -I %S/Inputs/template-default-args -std=c++11 %s -DBEGIN= -DEND=
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -verify -fmodules-cache-path=%t -fno-modules-error-recovery -I %S/Inputs/template-default-args -std=c++11 %s -DBEGIN="namespace N {" -DEND="}"

BEGIN
template<typename T> struct A;
template<typename T> struct B;
template<typename T> struct C;
template<typename T = int> struct D;
template<typename T = int> struct E {};
template<typename T> struct H {};
template<typename T = int, typename U = int> struct I {};
END

#include "b.h"
#include "d.h"

BEGIN
template<typename T = int> struct A {};
template<typename T> struct B {};
template<typename T = int> struct B;
template<typename T = int> struct C;
template<typename T> struct D {};
template<typename T> struct F {};
template<typename T> struct G {}; // #G
template<typename T> struct J {};
template<typename T = int> struct J;
struct K : J<> {};
END

#include "c.h"

BEGIN
A<> a;
B<> b;
extern C<> c;
D<> d;
E<> e;
F<> f;
G<> g; // expected-error {{missing '#include "a.h"'; default argument of 'G' must be defined before it is used}}
// expected-note@a.h:7 {{default argument declared here is not reachable}}
// expected-note@#G {{template parameter is declared here}}
H<> h; // expected-error {{missing '#include "a.h"'; default argument of 'H' must be defined before it is used}}
// expected-note@a.h:8 {{default argument declared here is not reachable}}
// expected-note@a.h:8 {{template parameter is declared here}}
I<> i;
L<> *l;
END

namespace DeferredLookup {
  template<typename T, typename U = T> using X = U;
  template<typename T> void f() { (void) X<T>(); }
  template<typename T> int n = X<T>(); // expected-warning {{extension}}
  template<typename T> struct S { X<T> xt; enum E : int; };
  template<typename T> enum S<T>::E : int { a = X<T>() };

  void test() {
    f<int>();
    n<int> = 1;
    S<int> s;
    S<int>::E e = S<int>::E::a;

    Indirect::B<int>::C<int> indirect;
  }
}
