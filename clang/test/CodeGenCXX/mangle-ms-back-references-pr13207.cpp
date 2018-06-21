// RUN: %clang_cc1 -emit-llvm %s -o - -triple=i386-pc-win32 | FileCheck %s

template<class X, class Y, class Z>
class A {};
template<class X>
class B {};
template<class X>
class C {};

void foo_abbb(A<B<char>, B<char>, B<char> >) {}
// CHECK: "\01?foo_abbb@@YAXV?$A@V?$B@D@@V1@V1@@@@Z"
void foo_abb(A<char, B<char>, B<char> >) {}
// CHECK: "\01?foo_abb@@YAXV?$A@DV?$B@D@@V1@@@@Z"
void foo_abc(A<char, B<char>, C<char> >) {}
// CHECK: "\01?foo_abc@@YAXV?$A@DV?$B@D@@V?$C@D@@@@@Z"
void foo_bt(bool a, B<bool(bool)> b) {}
// CHECK: "\01?foo_bt@@YAX_NV?$B@$$A6A_N_N@Z@@@Z"

namespace N {
template<class X, class Y, class Z>
class A {};
template<class X>
class B {};
template<class X>
class C {};
template<class X, class Y>
class D {};
class Z {};
}

void foo_abbb(N::A<N::B<char>, N::B<char>, N::B<char> >) {}
// CHECK: "\01?foo_abbb@@YAXV?$A@V?$B@D@N@@V12@V12@@N@@@Z"
void foo_abb(N::A<char, N::B<char>, N::B<char> >) {}
// CHECK: "\01?foo_abb@@YAXV?$A@DV?$B@D@N@@V12@@N@@@Z"
void foo_abc(N::A<char, N::B<char>, N::C<char> >) {}
// CHECK: "\01?foo_abc@@YAXV?$A@DV?$B@D@N@@V?$C@D@2@@N@@@Z"

N::A<char, N::B<char>, N::C<char> > abc_foo() {
// CHECK: ?abc_foo@@YA?AV?$A@DV?$B@D@N@@V?$C@D@2@@N@@XZ
  return N::A<char, N::B<char>, N::C<char> >();
}

N::Z z_foo(N::Z arg) {
// CHECK: ?z_foo@@YA?AVZ@N@@V12@@Z
  return arg;
}

N::B<char> b_foo(N::B<char> arg) {
// CHECK: ?b_foo@@YA?AV?$B@D@N@@V12@@Z
  return arg;
}

N::D<char, char> d_foo(N::D<char, char> arg) {
// CHECK: ?d_foo@@YA?AV?$D@DD@N@@V12@@Z
  return arg;
}

N::A<char, N::B<char>, N::C<char> > abc_foo_abc(N::A<char, N::B<char>, N::C<char> >) {
// CHECK: ?abc_foo_abc@@YA?AV?$A@DV?$B@D@N@@V?$C@D@2@@N@@V12@@Z
  return N::A<char, N::B<char>, N::C<char> >();
}

namespace NA {
class X {};
template<class T> class Y {};
}

namespace NB {
class X {};
template<class T> class Y {};
}

void foo5(NA::Y<NB::Y<NA::Y<NB::Y<NA::X> > > > arg) {}
// CHECK: "\01?foo5@@YAXV?$Y@V?$Y@V?$Y@V?$Y@VX@NA@@@NB@@@NA@@@NB@@@NA@@@Z"

void foo11(NA::Y<NA::X>, NB::Y<NA::X>) {}
// CHECK: "\01?foo11@@YAXV?$Y@VX@NA@@@NA@@V1NB@@@Z"

void foo112(NA::Y<NA::X>, NB::Y<NB::X>) {}
// CHECK: "\01?foo112@@YAXV?$Y@VX@NA@@@NA@@V?$Y@VX@NB@@@NB@@@Z"

void foo22(NA::Y<NB::Y<NA::X> >, NB::Y<NA::Y<NA::X> >) {}
// CHECK: "\01?foo22@@YAXV?$Y@V?$Y@VX@NA@@@NB@@@NA@@V?$Y@V?$Y@VX@NA@@@NA@@@NB@@@Z"

namespace PR13207 {
class A {};
class B {};
class C {};

template<class X>
class F {};
template<class X>
class I {};
template<class X, class Y>
class J {};
template<class X, class Y, class Z>
class K {};

class L {
 public:
  void foo(I<A> x) {}
};
// CHECK: "\01?foo@L@PR13207@@QAEXV?$I@VA@PR13207@@@2@@Z"

void call_l_foo(L* l) { l->foo(I<A>()); }

void foo(I<A> x) {}
// CHECK: "\01?foo@PR13207@@YAXV?$I@VA@PR13207@@@1@@Z"
void foo2(I<A> x, I<A> y) { }
// CHECK: "\01?foo2@PR13207@@YAXV?$I@VA@PR13207@@@1@0@Z"
void bar(J<A,B> x) {}
// CHECK: "\01?bar@PR13207@@YAXV?$J@VA@PR13207@@VB@2@@1@@Z"
void spam(K<A,B,C> x) {}
// CHECK: "\01?spam@PR13207@@YAXV?$K@VA@PR13207@@VB@2@VC@2@@1@@Z"

void baz(K<char, F<char>, I<char> >) {}
// CHECK: "\01?baz@PR13207@@YAXV?$K@DV?$F@D@PR13207@@V?$I@D@2@@1@@Z"
void qux(K<char, I<char>, I<char> >) {}
// CHECK: "\01?qux@PR13207@@YAXV?$K@DV?$I@D@PR13207@@V12@@1@@Z"

namespace NA {
class X {};
template<class T> class Y {};
void foo(Y<X> x) {}
// CHECK: "\01?foo@NA@PR13207@@YAXV?$Y@VX@NA@PR13207@@@12@@Z"
void foofoo(Y<Y<X> > x) {}
// CHECK: "\01?foofoo@NA@PR13207@@YAXV?$Y@V?$Y@VX@NA@PR13207@@@NA@PR13207@@@12@@Z"
}

namespace NB {
class X {};
template<class T> class Y {};
void foo(Y<NA::X> x) {}
// CHECK: "\01?foo@NB@PR13207@@YAXV?$Y@VX@NA@PR13207@@@12@@Z"

void bar(NA::Y<X> x) {}
// CHECK: "\01?bar@NB@PR13207@@YAXV?$Y@VX@NB@PR13207@@@NA@2@@Z"

void spam(NA::Y<NA::X> x) {}
// CHECK: "\01?spam@NB@PR13207@@YAXV?$Y@VX@NA@PR13207@@@NA@2@@Z"

void foobar(NA::Y<Y<X> > a, Y<Y<X> >) {}
// CHECK: "\01?foobar@NB@PR13207@@YAXV?$Y@V?$Y@VX@NB@PR13207@@@NB@PR13207@@@NA@2@V312@@Z"

void foobarspam(Y<X> a, NA::Y<Y<X> > b, Y<Y<X> >) {}
// CHECK: "\01?foobarspam@NB@PR13207@@YAXV?$Y@VX@NB@PR13207@@@12@V?$Y@V?$Y@VX@NB@PR13207@@@NB@PR13207@@@NA@2@V412@@Z"

void foobarbaz(Y<X> a, NA::Y<Y<X> > b, Y<Y<X> >, Y<Y<X> > c) {}
// CHECK: "\01?foobarbaz@NB@PR13207@@YAXV?$Y@VX@NB@PR13207@@@12@V?$Y@V?$Y@VX@NB@PR13207@@@NB@PR13207@@@NA@2@V412@2@Z"

void foobarbazqux(Y<X> a, NA::Y<Y<X> > b, Y<Y<X> >, Y<Y<X> > c , NA::Y<Y<Y<X> > > d) {}
// CHECK: "\01?foobarbazqux@NB@PR13207@@YAXV?$Y@VX@NB@PR13207@@@12@V?$Y@V?$Y@VX@NB@PR13207@@@NB@PR13207@@@NA@2@V412@2V?$Y@V?$Y@V?$Y@VX@NB@PR13207@@@NB@PR13207@@@NB@PR13207@@@52@@Z"
}

namespace NC {
class X {};
template<class T> class Y {};

void foo(Y<NB::X> x) {}
// CHECK: "\01?foo@NC@PR13207@@YAXV?$Y@VX@NB@PR13207@@@12@@Z"

void foobar(NC::Y<NB::Y<NA::Y<NA::X> > > x) {}
// CHECK: "\01?foobar@NC@PR13207@@YAXV?$Y@V?$Y@V?$Y@VX@NA@PR13207@@@NA@PR13207@@@NB@PR13207@@@12@@Z"
}
}

// Function template names are not considered for backreferencing, but normal
// function names are.
namespace fn_space {
struct RetVal { int hash; };
template <typename T>
RetVal fun_tmpl(const T &t) { return RetVal(); }
RetVal fun_normal(int t) { return RetVal(); }
void fun_instantiate() {
  fun_normal(1);
  fun_tmpl(1);
}
// CHECK: "\01?fun_normal@fn_space@@YA?AURetVal@1@H@Z"
// CHECK: "\01??$fun_tmpl@H@fn_space@@YA?AURetVal@0@ABH@Z"

template <typename T, RetVal (*F)(T)>
RetVal fun_tmpl_recurse(T t) {
  if (!t)
    return RetVal();
  return F(t - 1);
}
RetVal ident(int x) { return RetVal(); }
void fun_instantiate2() {
  fun_tmpl_recurse<int, fun_tmpl_recurse<int, ident> >(10);
}
// CHECK: "\01??$fun_tmpl_recurse@H$1??$fun_tmpl_recurse@H$1?ident@fn_space@@YA?AURetVal@2@H@Z@fn_space@@YA?AURetVal@1@H@Z@fn_space@@YA?AURetVal@0@H@Z"
// CHECK: "\01??$fun_tmpl_recurse@H$1?ident@fn_space@@YA?AURetVal@2@H@Z@fn_space@@YA?AURetVal@0@H@Z"
}
