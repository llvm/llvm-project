// RUN: %clang_cc1 -no-opaque-pointers %s -emit-llvm -triple %itanium_abi_triple -std=c++11 -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers %s -emit-llvm -triple i686-linux-gnu -std=c++11 -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers %s -emit-llvm -triple x86_64-linux-gnu -std=c++11 -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers %s -emit-llvm -triple powerpc64le-unknown-linux-gnu -std=c++11 -o - | FileCheck %s

struct __attribute__((abi_tag("A", "B"))) A { };

struct B: A { };

template<class T>

struct C {
};

struct D { A* p; };

template<class T>
struct __attribute__((abi_tag("C", "D"))) E {
};

struct __attribute__((abi_tag("A", "B"))) F { };

A a1;
// CHECK-DAG: @_Z2a1B1AB1B =

__attribute__((abi_tag("C", "D")))
A a2;
// CHECK-DAG: @_Z2a2B1AB1BB1CB1D =

B a3;
// CHECK-DAG: @a3 =

C<A> a4;
// CHECK-DAG: @_Z2a4B1AB1B =

D a5;
// CHECK-DAG: @a5 =

E<int> a6;
// CHECK-DAG: @_Z2a6B1CB1D =

E<A> a7;
// CHECK-DAG: @_Z2a7B1AB1BB1CB1D =

template<>
struct E<float> {
  static float a8;
};
float E<float>::a8;
// CHECK-DAG: @_ZN1EB1CB1DIfE2a8E =

template<>
struct E<F> {
  static bool a9;
};
bool E<F>::a9;
// CHECK-DAG: @_ZN1EB1CB1DI1FB1AB1BE2a9E =

struct __attribute__((abi_tag("A", "B"))) A10 {
  virtual ~A10() {}
} a10;
// vtable
// CHECK-DAG: @_ZTV3A10B1AB1B =
// typeinfo
// CHECK-DAG: @_ZTI3A10B1AB1B =

struct __attribute__((abi_tag("A"))) B11 {
  static A10 b;
};
A10 B11::b;
// B11[abi:A]::b[abi:B]
// CHECK-DAG: @_ZN3B11B1A1bB1BE =

__attribute__ ((abi_tag("C", "D")))
void* f1() {
  return 0;
}
// CHECK-DAG: define {{.*}} @_Z2f1B1CB1Dv(

__attribute__ ((abi_tag("C", "D")))
A* f2() {
  return 0;
}
// CHECK-DAG: define {{.*}} @_Z2f2B1AB1BB1CB1Dv(

B* f3() {
  return 0;
}
// CHECK-DAG: define {{.*}} @_Z2f3v(

C<A>* f4() {
  return 0;
}
// CHECK-DAG: define {{.*}} @_Z2f4B1AB1Bv(

D* f5() {
  return 0;
}
// CHECK-DAG: define {{.*}} @_Z2f5v(

E<char>* f6() {
  return 0;
}
// CHECK-DAG: define {{.*}} @_Z2f6B1CB1Dv(

E<A>* f7() {
  return 0;
}
// CHECK-DAG: define {{.*}} @_Z2f7B1AB1BB1CB1Dv(

void f8(E<A>*) {
}
// CHECK-DAG: define {{.*}} @_Z2f8P1EB1CB1DI1AB1AB1BE(

inline namespace Names1 __attribute__((__abi_tag__)) {
    class C1 {};
}
C1 f9() { return C1(); }
// CHECK-DAG: @_Z2f9B6Names1v(

inline namespace Names2 __attribute__((__abi_tag__("Tag1", "Tag2"))) {
    class C2 {};
}
C2 f10() { return C2(); }
// CHECK-DAG: @_Z3f10B4Tag1B4Tag2v(

void __attribute__((abi_tag("A"))) f11(A) {}
// f11[abi:A](A[abi:A][abi:B])
// CHECK-DAG: define {{.*}} @_Z3f11B1A1AB1AB1B(

A f12(A) { return A(); }
// f12(A[abi:A][abi:B])
// CHECK-DAG: define {{.*}} @_Z3f121AB1AB1B(

inline void f13() {
  struct L {
    static E<int>* foo() {
      static A10 a;
      return 0;
    }
  };
  L::foo();
}
void f13_test() {
  f13();
}
// f13()::L::foo[abi:C][abi:D]()
// CHECK-DAG: define linkonce_odr {{(dso_local )?}}{{(noundef )?}}%struct.E* @_ZZ3f13vEN1L3fooB1CB1DEv(

// f13()::L::foo[abi:C][abi:D]()::a[abi:A][abi:B]
// CHECK-DAG: @_ZZZ3f13vEN1L3fooB1CB1DEvE1aB1AB1B =

// guard variable for f13()::L::foo[abi:C][abi:D]()::a[abi:A][abi:B]
// CHECK-DAG: @_ZGVZZ3f13vEN1L3fooB1CB1DEvE1aB1AB1B =

struct __attribute__((abi_tag("TAG"))) A14 {
  A14 f14();
};
A14 A14::f14() {
  return A14();
}
// A14[abi:TAG]::f14()
// CHECK-DAG: define {{.+}} @_ZN3A14B3TAG3f14Ev(

template<class T>
T f15() {
  return T();
}
void f15_test() {
  f15<A14>();
}
// A14[abi:TAG] f15<A14[abi:TAG]>()
// CHECK-DAG: define linkonce_odr {{.+}} @_Z3f15I3A14B3TAGET_v(

template<class T>
A14 f16() {
  return A14();
}
void f16_test() {
  f16<int>();
}
// A14[abi:TAG] f16<int>()
// CHECK-DAG: define linkonce_odr {{.+}} @_Z3f16IiE3A14B3TAGv(

template<class T>
struct __attribute__((abi_tag("TAG"))) A17 {
  A17 operator+(const A17& a) {
    return a;
  }
};
void f17_test() {
  A17<int> a, b;
  a + b;
}
// A17[abi:TAG]<int>::operator+(A17[abi:TAG]<int> const&)
// CHECK-DAG: define linkonce_odr {{.+}} @_ZN3A17B3TAGIiEplERKS0_(

struct A18 {
  operator A() { return A(); }
};
void f18_test() {
  A a = A18();
}
// A18::operator A[abi:A][abi:B]() but GCC adds the same tags twice!
// CHECK-DAG: define linkonce_odr {{.+}} @_ZN3A18cv1AB1AB1BEv(

namespace N19 {
  class A {};
  class __attribute__((abi_tag("B"))) B {};
  class D {};
  class F {};

  template<typename T, B F(T, D)>
  class C {};

  B foo(A, D);
}
void f19_test(N19::C<N19::A,  &N19::foo>, N19::F, N19::D) {
}
// f19_test(N19::C<N19::A, &N19::foo[abi:B]>, N19::F, N19::D)
// CHECK-DAG: define {{(dso_local )?}}void @_Z8f19_testN3N191CINS_1AEXadL_ZNS_3fooB1BES1_NS_1DEEEEENS_1FES2_(

namespace pr30440 {

template<class F> void g(F);
template<class ...A> auto h(A ...a)->decltype (g (0, g < a > (a) ...)) {
}
// CHECK-DAG: define {{.*}} @_ZN7pr304401hIJEEEDTcl1gLi0Espcl1gIXfp_EEfp_EEEDpT_(

void pr30440_test () {
  h();
}

}
