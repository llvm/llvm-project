// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-apple-darwin9 -fblocks -std=c++11 | FileCheck %s
struct X { };
struct Y { };

// CHECK: @unmangled_variable = global
// CHECK: @_ZN1N1iE = global
// CHECK: @_ZZN1N1fEiiE1b = internal global
// CHECK: @_ZZN1N1gEvE1a = internal global
// CHECK: @_ZGVZN1N1gEvE1a = internal global

//CHECK: @pr5966_i = external global
//CHECK: @_ZL8pr5966_i = internal global

// CHECK: define zeroext i1 @_ZplRK1YRA100_P1X
bool operator+(const Y&, X* (&xs)[100]) { return false; }

// CHECK: define void @_Z1f1s
typedef struct { int a; } s;
void f(s) { }

// CHECK: define void @_Z1f1e
typedef enum { foo } e;
void f(e) { }

// CHECK: define void @_Z1f1u
typedef union { int a; } u;
void f(u) { }

// CHECK: define void @_Z1f1x
typedef struct { int a; } x,y;
void f(y) { }

// CHECK: define void @_Z1fv
void f() { }

// CHECK: define void @_ZN1N1fEv
namespace N { void f() { } }

// CHECK: define void @_ZN1N1N1fEv
namespace N { namespace N { void f() { } } }

// CHECK: define void @unmangled_function
extern "C" { namespace N { void unmangled_function() { } } }

extern "C" { namespace N { int unmangled_variable = 10; } }

namespace N { int i; }

namespace N { int f(int, int) { static int b; return b; } }

namespace N { int h(); void g() { static int a = h(); } }

// CHECK: define void @_Z1fno
void f(__int128_t, __uint128_t) { } 

template <typename T> struct S1 {};

// CHECK: define void @_Z1f2S1IiE
void f(S1<int>) {}

// CHECK: define void @_Z1f2S1IdE
void f(S1<double>) {}

template <int N> struct S2 {};
// CHECK: define void @_Z1f2S2ILi100EE
void f(S2<100>) {}

// CHECK: define void @_Z1f2S2ILin100EE
void f(S2<-100>) {}

template <bool B> struct S3 {};

// CHECK: define void @_Z1f2S3ILb1EE
void f(S3<true>) {}

// CHECK: define void @_Z1f2S3ILb0EE
void f(S3<false>) {}

struct S;

// CHECK: define void @_Z1fM1SKFvvE
void f(void (S::*)() const) {}

// CHECK: define void @_Z1fM1SFvvE
void f(void (S::*)()) {}

// CHECK: define void @_Z1fi
void f(const int) { }

template<typename T, typename U> void ft1(U u, T t) { }

template<typename T> void ft2(T t, void (*)(T), void (*)(T)) { }

template<typename T, typename U = S1<T> > struct S4 { };
template<typename T> void ft3(S4<T>*) {  }

namespace NS {
  template<typename T> void ft1(T) { }
}

void g1() {
  // CHECK: @_Z3ft1IidEvT0_T_
  ft1<int, double>(1, 0);
  
  // CHECK: @_Z3ft2IcEvT_PFvS0_ES2_
  ft2<char>(1, 0, 0);
  
  // CHECK: @_Z3ft3IiEvP2S4IT_2S1IS1_EE
  ft3<int>(0);
  
  // CHECK: @_ZN2NS3ft1IiEEvT_
  NS::ft1<int>(1);
}

// Expressions
template<int I> struct S5 { };

template<int I> void ft4(S5<I>) { }
void g2() {
  // CHECK: @_Z3ft4ILi10EEv2S5IXT_EE
  ft4(S5<10>());
  
  // CHECK: @_Z3ft4ILi20EEv2S5IXT_EE
  ft4(S5<20>());
}

extern "C++" {
  // CHECK: @_Z1hv
 void h() { } 
}

// PR5019
extern "C" { struct a { int b; }; }

// CHECK: @_Z1fP1a
int f(struct a *x) {
    return x->b;
}

// PR5017
extern "C" {
struct Debug {
  const Debug& operator<< (unsigned a) const { return *this; }
};
Debug dbg;
// CHECK: @_ZNK5DebuglsEj
int main(void) {  dbg << 32 ;}
}

template<typename T> struct S6 {
  typedef int B;
};

template<typename T> void ft5(typename S6<T>::B) { }
// CHECK: @_Z3ft5IiEvN2S6IT_E1BE
template void ft5<int>(int);

template<typename T> class A {};

namespace NS {
template<typename T> bool operator==(const A<T>&, const A<T>&) { return true; }
}

// CHECK: @_ZN2NSeqIcEEbRK1AIT_ES5_
template bool NS::operator==(const ::A<char>&, const ::A<char>&);

namespace std {
template<typename T> bool operator==(const A<T>&, const A<T>&) { return true; }
}

// CHECK: @_ZSteqIcEbRK1AIT_ES4_
template bool std::operator==(const ::A<char>&, const ::A<char>&);

struct S {
  typedef int U;
};

template <typename T> typename T::U ft6(const T&) { return 0; }

// CHECK: @_Z3ft6I1SENT_1UERKS1_
template int ft6<S>(const S&);

template<typename> struct __is_scalar_type {
  enum { __value = 1 };
};

template<bool, typename> struct __enable_if { };

template<typename T> struct __enable_if<true, T> {
  typedef T __type;
};

// PR5063
template<typename T> typename __enable_if<__is_scalar_type<T>::__value, void>::__type ft7() { }

// CHECK: @_Z3ft7IiEN11__enable_ifIXsr16__is_scalar_typeIT_EE7__valueEvE6__typeEv
template void ft7<int>();
// CHECK: @_Z3ft7IPvEN11__enable_ifIXsr16__is_scalar_typeIT_EE7__valueEvE6__typeEv
template void ft7<void*>();

// PR5144
extern "C" {
void extern_f(void);
};

// CHECK: @extern_f
void extern_f(void) { }

struct S7 {
  S7();
  
  struct S { S(); };
  struct {
    S s;
  } a;
};

// PR5139
// CHECK: @_ZN2S7C1Ev
// CHECK: @_ZN2S7C2Ev
// CHECK: @"_ZN2S73$_0C1Ev"
S7::S7() {}

// PR5063
template<typename T> typename __enable_if<(__is_scalar_type<T>::__value), void>::__type ft8() { }
// CHECK: @_Z3ft8IiEN11__enable_ifIXsr16__is_scalar_typeIT_EE7__valueEvE6__typeEv
template void ft8<int>();
// CHECK: @_Z3ft8IPvEN11__enable_ifIXsr16__is_scalar_typeIT_EE7__valueEvE6__typeEv
template void ft8<void*>();

// PR5796
namespace PR5796 {
template<typename> struct __is_scalar_type {
  enum { __value = 0 };
};

template<bool, typename> struct __enable_if {};
template<typename T> struct __enable_if<true, T> { typedef T __type; };
template<typename T>

// CHECK: define linkonce_odr void @_ZN6PR57968__fill_aIiEENS_11__enable_ifIXntsr16__is_scalar_typeIT_EE7__valueEvE6__typeEv
typename __enable_if<!__is_scalar_type<T>::__value, void>::__type __fill_a() { };

void f() { __fill_a<int>(); }
}

namespace Expressions {
// Unary operators.

// CHECK: define weak_odr void @_ZN11Expressions2f1ILi1EEEvPAplngT_Li2E_i
template <int i> void f1(int (*)[(-i) + 2]) { };
template void f1<1>(int (*)[1]);

// CHECK: define weak_odr void @_ZN11Expressions2f2ILi1EEEvPApsT__i
template <int i> void f2(int (*)[+i]) { };
template void f2<1>(int (*)[1]);

// Binary operators.

// CHECK: define weak_odr void @_ZN11Expressions2f3ILi1EEEvPAplT_T__i
template <int i> void f3(int (*)[i+i]) { };
template void f3<1>(int (*)[2]);

// CHECK: define weak_odr void @_ZN11Expressions2f4ILi1EEEvPAplplLi2ET_T__i
template <int i> void f4(int (*)[2 + i+i]) { };
template void f4<1>(int (*)[4]);

// The ternary operator.
// CHECK: define weak_odr void @_ZN11Expressions2f4ILb1EEEvPAquT_Li1ELi2E_i
template <bool b> void f4(int (*)[b ? 1 : 2]) { };
template void f4<true>(int (*)[1]);
}

struct Ops {
  Ops& operator+(const Ops&);
  Ops& operator-(const Ops&);
  Ops& operator&(const Ops&);
  Ops& operator*(const Ops&);
  
  void *v;
};

// CHECK: define %struct.Ops* @_ZN3OpsplERKS_
Ops& Ops::operator+(const Ops&) { return *this; }
// CHECK: define %struct.Ops* @_ZN3OpsmiERKS_
Ops& Ops::operator-(const Ops&) { return *this; }
// CHECK: define %struct.Ops* @_ZN3OpsanERKS_
Ops& Ops::operator&(const Ops&) { return *this; }
// CHECK: define %struct.Ops* @_ZN3OpsmlERKS_
Ops& Ops::operator*(const Ops&) { return *this; }

// PR5861
namespace PR5861 {
template<bool> class P;
template<> class P<true> {};

template<template <bool> class, bool>
struct Policy { };

template<typename T, typename = Policy<P, true> > class Alloc
{
  T *allocate(int, const void*) { return 0; }
};

// CHECK: define weak_odr i8* @_ZN6PR58615AllocIcNS_6PolicyINS_1PELb1EEEE8allocateEiPKv
template class Alloc<char>;
}

// CHECK: define void @_Z1fU13block_pointerFiiiE
void f(int (^)(int, int)) { }

void pr5966_foo() {
  extern int pr5966_i;
  pr5966_i = 0;
}

static int pr5966_i;

void pr5966_bar() {
  pr5966_i = 0;
}

namespace test0 {
  int ovl(int x);
  char ovl(double x);

  template <class T> void f(T, char (&buffer)[sizeof(ovl(T()))]) {}

  void test0() {
    char buffer[1];
    f(0.0, buffer);
  }
  // CHECK: define void @_ZN5test05test0Ev()
  // CHECK: define linkonce_odr void @_ZN5test01fIdEEvT_RAszcl3ovlcvS1__EE_c(

  void test1() {
    char buffer[sizeof(int)];
    f(1, buffer);
  }
  // CHECK: define void @_ZN5test05test1Ev()
  // CHECK: define linkonce_odr void @_ZN5test01fIiEEvT_RAszcl3ovlcvS1__EE_c(

  template <class T> void g(char (&buffer)[sizeof(T() + 5.0f)]) {}
  void test2() {
    char buffer[sizeof(float)];
    g<float>(buffer);
  }
  // CHECK: define linkonce_odr void @_ZN5test01gIfEEvRAszplcvT__ELf40a00000E_c(

  template <class T> void h(char (&buffer)[sizeof(T() + 5.0)]) {}
  void test3() {
    char buffer[sizeof(double)];
    h<float>(buffer);
  }
  // CHECK: define linkonce_odr void @_ZN5test01hIfEEvRAszplcvT__ELd4014000000000000E_c(

  template <class T> void j(char (&buffer)[sizeof(T().buffer)]) {}
  struct A { double buffer[128]; };
  void test4() {
    char buffer[1024];
    j<A>(buffer);
  }
  // CHECK: define linkonce_odr void @_ZN5test01jINS_1AEEEvRAszdtcvT__E6buffer_c(

  template <class T> void k(char (&buffer)[sizeof(T() + 0.0f)]) {}
  void test5() {
    char buffer[sizeof(float)];
    k<float>(buffer);
  }
  // CHECK: define linkonce_odr void @_ZN5test01kIfEEvRAszplcvT__ELf00000000E_c(

}

namespace test1 {
  template<typename T> struct X { };
  template<template<class> class Y, typename T> void f(Y<T>) { }
  // CHECK: define weak_odr void @_ZN5test11fINS_1XEiEEvT_IT0_E
  template void f(X<int>);
}

// CHECK: define internal void @_ZL27functionWithInternalLinkagev()
static void functionWithInternalLinkage() {  }
void g() { functionWithInternalLinkage(); }

namespace test2 {
  template <class T> decltype(((T*) 0)->member) read_member(T& obj) {
    return obj.member;
  }

  struct A { int member; } obj;
  int test() {
    return read_member(obj);
  }

  // CHECK: define linkonce_odr i32 @_ZN5test211read_memberINS_1AEEEDtptcvPT_Li0E6memberERS2_(
}

// rdar://problem/9280586
namespace test3 {
  struct AmbiguousBase { int ab; };
  struct Path1 : AmbiguousBase { float p; };
  struct Path2 : AmbiguousBase { double p; };
  struct Derived : Path1, Path2 { };

  // CHECK: define linkonce_odr i32 @_ZN5test38get_ab_1INS_7DerivedEEEDtptcvPT_Li0Esr5Path1E2abERS2_(
  template <class T> decltype(((T*) 0)->Path1::ab) get_ab_1(T &ref) { return ref.Path1::ab; }

  // CHECK: define linkonce_odr i32 @_ZN5test38get_ab_2INS_7DerivedEEEDtptcvPT_Li0Esr5Path2E2abERS2_(
  template <class T> decltype(((T*) 0)->Path2::ab) get_ab_2(T &ref) { return ref.Path2::ab; }

  // CHECK: define linkonce_odr float @_ZN5test37get_p_1INS_7DerivedEEEDtptcvPT_Li0Esr5Path1E1pERS2_(
  template <class T> decltype(((T*) 0)->Path1::p) get_p_1(T &ref) { return ref.Path1::p; }

  // CHECK: define linkonce_odr double @_ZN5test37get_p_2INS_7DerivedEEEDtptcvPT_Li0Esr5Path2E1pERS2_(
  template <class T> decltype(((T*) 0)->Path2::p) get_p_2(T &ref) { return ref.Path2::p; }

  Derived obj;
  void test() {
    get_ab_1(obj);
    get_ab_2(obj);
    get_p_1(obj);
    get_p_2(obj);
  }
}

// CHECK: define void @_ZN5test41gEPNS_3zedIXadL_ZNS_3foo3barEEEEE
namespace test4 {
  struct foo { int bar; };
  template <int (foo::*)>
  struct zed {};
  void g(zed<&foo::bar>*)
  {}
}
// CHECK: define void @_ZN5test51gEPNS_3zedIXadL_ZNS_3foo3barEEEEE
namespace test5 {
  struct foo { static int bar; };
  template <int *>
  struct zed {};
  void g(zed<&foo::bar>*)
  {}
}
// CHECK: define void @_ZN5test61gEPNS_3zedIXadL_ZNS_3foo3barEvEEEE
namespace test6 {
  struct foo { int bar(); };
  template <int (foo::*)()>
  struct zed {};
  void g(zed<&foo::bar>*)
  {}
}
// CHECK: define void @_ZN5test71gEPNS_3zedIXadL_ZNS_3foo3barEvEEEE
namespace test7 {
  struct foo { static int bar(); };
  template <int (*f)()>
  struct zed {};
  void g(zed<&foo::bar>*)
  {}
}
// CHECK: define weak_odr void @_ZN5test81AILZNS_1B5valueEEE3incEv
namespace test8 {
  template <int &counter> class A { void inc() { counter++; } };
  class B { public: static int value; };
  template class A<B::value>;
}
// CHECK: declare void @_ZN5test91fIiNS_3barEEEvRKNT0_3baz1XE
namespace test9 {
  template<class T>
  struct foo {
    typedef T X;
  };
  struct bar {
    typedef foo<int> baz;
  };
  template <class zaz, class zed>
  void f(const typename zed::baz::X&);
  void g() {
    f<int, bar>( 0);
  }
}

// <rdar://problem/7825453>
namespace test10 {
  template <char P1> struct S {};
  template <char P2> void f(struct S<false ? 'a' : P2> ) {}

  // CHECK: define weak_odr void @_ZN6test101fILc3EEEvNS_1SIXquLb0ELc97ET_EEE(
  template void f<(char) 3>(struct S<3>);
}

namespace test11 {
  // CHECK: @_ZN6test111fEz
  void f(...) { }

  struct A {
    void f(...);
  };
  
  // CHECK: @_ZN6test111A1fEz
  void A::f(...) { }
}

namespace test12 {

  // CHECK: _ZN6test121fENS_1AILt33000EEE
  template <unsigned short> struct A { };
  void f(A<33000>) { }
}

// PR7446
namespace test13 {
  template <template <class> class T> class A {};
  template <class U> class B {};

  template <template<class> class T> void foo(const A<T> &a) {}

  // CHECK: define weak_odr void @_ZN6test133fooINS_1BEEEvRKNS_1AIT_EE(
  template void foo(const A<B> &a);
}

namespace test14 {
  extern "C" {
    struct S {
      static int a(), x;
    };
    // CHECK: define i32 @_ZN6test141S1aEv
    // CHECK: load i32* @_ZN6test141S1xE
    int S::a() { return S::x; }
  }
}

// rdar://problem/8204122
namespace test15 {
  enum E { e = 3 };
  template <int I> struct S {};

  template <int I> void f(S<I + e>) {}

  // CHECK: define weak_odr void @_ZN6test151fILi7EEEvNS_1SIXplT_LNS_1EE3EEEE(
  template void f<7>(S<7 + e>);
}

// rdar://problem/8302148
namespace test17 {
  template <int N> struct A {};

  struct B {
    static int foo(void);
  };

  template <class T> A<sizeof(T::foo())> func(void);

  // CHECK: define void @_ZN6test174testEv()
  // CHECK: call {{.*}} @_ZN6test174funcINS_1BEEENS_1AIXszclsrT_3fooEEEEv()
  void test() {
    func<B>();
  }
}

// PR7891
namespace test18 {
  struct A {
    int operator+();
    int operator-();
    int operator*();
    int operator&();
  };
  template <int (A::*)()> struct S {};

  template <typename T> void f(S<&T::operator+>) {}
  template void f<A>(S<&A::operator+>);

  template <typename T> void f(S<&T::operator- >) {}
  template void f<A>(S<&A::operator- >);

  template <typename T> void f(S<&T::operator*>) {}
  template void f<A>(S<&A::operator*>);

  template <typename T> void f(S<&T::operator&>) {}
  template void f<A>(S<&A::operator&>);

  // CHECK: define weak_odr void @_ZN6test181fINS_1AEEEvNS_1SIXadsrT_plEEE
  // CHECK: define weak_odr void @_ZN6test181fINS_1AEEEvNS_1SIXadsrT_miEEE
  // CHECK: define weak_odr void @_ZN6test181fINS_1AEEEvNS_1SIXadsrT_mlEEE
  // CHECK: define weak_odr void @_ZN6test181fINS_1AEEEvNS_1SIXadsrT_anEEE
}

// rdar://problem/8332117
namespace test19 {
  struct A {
    template <typename T> int f();
    int operator+();
    operator int();
    template <typename T> int operator-();
  };

  template <int (A::*)()> struct S {};

  template <typename T> void g (S<&T::template f<int> >) {}
  template <typename T> void g (S<&T::operator+ >) {}
  template <typename T> void g (S<&T::operator int>) {}
  template <typename T> void g (S<&T::template operator- <double> >) {}

  // CHECK: define weak_odr void @_ZN6test191gINS_1AEEEvNS_1SIXadsrT_1fIiEEEE(
  template void g<A>(S<&A::f<int> >);
  // CHECK: define weak_odr void @_ZN6test191gINS_1AEEEvNS_1SIXadsrT_plEEE(
  template void g<A>(S<&A::operator+>);
  // CHECK: define weak_odr void @_ZN6test191gINS_1AEEEvNS_1SIXadsrT_cviEEE(
  template void g<A>(S<&A::operator int>);
  // CHECK: define weak_odr void @_ZN6test191gINS_1AEEEvNS_1SIXadsrT_miIdEEEE(
  template void g<A>(S<&A::operator-<double> >);
}

namespace test20 {
  template <class T> T *f(const T&);
  template <class T> T *f(T*);

  // CHECK: define weak_odr void @_ZN6test205test0IiEEvDTcl1fIPT_ELi0EEE(
  template <class T> void test0(decltype(f<T*>(0))) {}
  template void test0<int>(decltype(f<int*>(0)));

  // CHECK: define weak_odr void @_ZN6test205test1IiEEvDTcl1fIEcvT__EEE(
  template <class T> void test1(decltype(f<>(T()))) {}
  template void test1<int>(decltype(f<>(int())));
}

// rdar:// 8620510
namespace test21 {
  // CHECK: define void @_ZN6test2112vla_arg_funcEiPA_i(
  void vla_arg_func(int X, int a[X][X]) {}
}

namespace test22 {
  // CHECK: define void @_ZN6test221fEDn(
  void f(decltype(nullptr)) { }
}

// rdar://problem/8913416
namespace test23 {
  typedef void * const vpc;

  // CHECK: define void @_ZN6test231fERA10_KPv(
  void f(vpc (&)[10]) {}

  typedef vpc vpca5[5];
  void f(vpca5 volatile (&)[10]) {}
  // CHECK: define void @_ZN6test231fERA10_A5_VKPv(
}

namespace test24 {
  void test0() {
    extern int foo();
    // CHECK: call i32 @_ZN6test243fooEv()
    foo();
  }

  static char foo() {}
  void test1() {
    // CHECK: call signext i8 @_ZN6test24L3fooEv()
    foo();
  }
}

// rdar://problem/8806641
namespace test25 {
  template <void (*fn)()> struct A {
    static void call() { fn(); }
  };
  void foo();
  void test() {
    // CHECK: call void @_ZN6test251AIXadL_ZNS_3fooEvEEE4callEv()
    A<foo>::call();
  }
}

namespace test26 {
  template <template <class> class T> void foo(decltype(T<float>::object) &object) {}

  template <class T> struct holder { static T object; };

  void test() {
    float f;

    // CHECK: call void @_ZN6test263fooINS_6holderEEEvRDtsrT_IfE6objectE(
    foo<holder>(f);
  }
}

namespace test27 {
  struct A {
    struct inner {
      float object;
    };

    float meth();
  };
  typedef A Alias;

  template <class T> void a(decltype(T::inner::object) &object) {}
  template <class T> void b(decltype(T().Alias::meth()) &object) {}

  void test() {
    float f;
    // CHECK: call void @_ZN6test271aINS_1AEEEvRDtsrNT_5innerE6objectE(
    a<A>(f);
    // CHECK: call void @_ZN6test271bINS_1AEEEvRDTcldtcvT__Esr5AliasE4methEE(
    b<A>(f);
  }
}

// An injected class name type in a unresolved-name.
namespace test28 {
  template <class T> struct A {
    enum { bit };
  };

  template <class T> void foo(decltype(A<T>::A::bit) x);

  void test() {
    foo<char>(A<char>::bit);
    // CHECK: call void @_ZN6test283fooIcEEvDtsr1AIT_E1AE3bitE(
  }
}

// An enclosing template type parameter in an unresolved-name.
namespace test29 {
  template <class T> struct A {
    template <class U> static void foo(decltype(T::fn(U())) x);
  };
  struct B { static int fn(int); static long fn(long); };

  void test() {
    A<B>::foo<int>(0);
    // CHECK: call void @_ZN6test291AINS_1BEE3fooIiEEvDTclsrS1_2fncvT__EEE(
  }
}

// An enclosing template template parameter in an unresolved-name.
namespace test30 {
  template <template <class> class T> struct A {
    template <class U> static void foo(decltype(T<U>::fn()) x);
  };
  template <class T> struct B { static T fn(); };

  void test() {
    A<B>::foo<int>(0);
    // CHECK: call void @_ZN6test301AINS_1BEE3fooIiEEvDTclsrS1_IT_EE2fnEE(
  }
}

namespace test31 { // instantiation-dependent mangling of decltype
  int x;
  template<class T> auto f1(T p)->decltype(x) { return 0; }
  // The return type in the mangling of the template signature
  // is encoded as "i".
  template<class T> auto f2(T p)->decltype(p) { return 0; }
  // The return type in the mangling of the template signature
  // is encoded as "Dtfp_E".
  void g(int);
  template<class T> auto f3(T p)->decltype(g(p)) {}

  // CHECK: define weak_odr i32 @_ZN6test312f1IiEEiT_(
  template int f1(int);
  // CHECK: define weak_odr i32 @_ZN6test312f2IiEEDtfp_ET_
  template int f2(int);
  // CHECK: define weak_odr void @_ZN6test312f3IiEEDTcl1gfp_EET_
  template void f3(int);
}

// PR10205
namespace test32 {
  template<typename T, int=T::value> struct A {
    typedef int type;
  };
  struct B { enum { value = 4 }; };

  template <class T> typename A<T>::type foo() { return 0; }
  void test() {
    foo<B>();
    // CHECK: call i32 @_ZN6test323fooINS_1BEEENS_1AIT_XsrS3_5valueEE4typeEv()
  }
}

namespace test33 {
  template <class T> struct X {
    enum { value = T::value };
  };

  template<typename T, int=X<T>::value> struct A {
    typedef int type;
  };
  struct B { enum { value = 4 }; };

  template <class T> typename A<T>::type foo() { return 0; }

  void test() {
    foo<B>();
    // CHECK: call i32 @_ZN6test333fooINS_1BEEENS_1AIT_Xsr1XIS3_EE5valueEE4typeEv()
  }
}

namespace test34 {
  // Mangling for instantiation-dependent decltype expressions.
  template<typename T>
  void f(decltype(sizeof(decltype(T() + T())))) {}

  // CHECK: define weak_odr void @_ZN6test341fIiEEvDTstDTplcvT__EcvS1__EEE
  template void f<int>(decltype(sizeof(1)));

  // Mangling for non-instantiation-dependent sizeof expressions.
  template<unsigned N>
  void f2(int (&)[N + sizeof(int*)]) {}

  // CHECK: define weak_odr void @_ZN6test342f2ILj4EEEvRAplT_Lm8E_i
  template void f2<4>(int (&)[4 + sizeof(int*)]);

  // Mangling for non-instantiation-dependent sizeof expressions
  // involving an implicit conversion of the result of the sizeof.
  template<unsigned long long N>
  void f3(int (&)[N + sizeof(int*)]) {}

  // CHECK: define weak_odr void @_ZN6test342f3ILy4EEEvRAplT_Ly8E_i
  template void f3<4>(int (&)[4 + sizeof(int*)]);

  // Mangling for instantiation-dependent sizeof() expressions as
  // template arguments.
  template<unsigned> struct A { };

  template<typename T> void f4(::test34::A<sizeof(sizeof(decltype(T() + T())))>) { }

  // CHECK: define weak_odr void @_ZN6test342f4IiEEvNS_1AIXszstDTplcvT__EcvS2__EEEEE
  template void f4<int>(A<sizeof(sizeof(int))>);
}

namespace test35 {
  // Dependent operator names of unknown arity.
  struct A { 
    template<typename U> A operator+(U) const;
  };

  template<typename T>
  void f1(decltype(sizeof(&T::template operator+<int>))) {}

  // CHECK: define weak_odr void @_ZN6test352f1INS_1AEEEvDTszadsrT_plIiEE
  template void f1<A>(__SIZE_TYPE__);
}

namespace test36 {
  template<unsigned> struct A { };

  template<typename ...Types>
  auto f1(Types... values) -> A<sizeof...(values)> { }

  // CHECK: define weak_odr {{.*}} @_ZN6test362f1IJifEEENS_1AIXsZfp_EEEDpT_
  template A<2> f1(int, float);
}
