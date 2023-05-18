// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-apple-darwin9 | FileCheck %s

struct X {};

// CHECK-LABEL: define{{.*}} void @_Z1f1XS_(
void f(X, X) { }

// CHECK-LABEL: define{{.*}} void @_Z1fR1XS0_(
void f(X&, X&) { }

// CHECK-LABEL: define{{.*}} void @_Z1fRK1XS1_(
void f(const X&, const X&) { }

typedef void T();
struct S {};

// CHECK-LABEL: define{{.*}} void @_Z1fPFvvEM1SFvvE(
void f(T*, T (S::*)) {}

namespace A {
  struct A { };
  struct B { };
};

// CHECK-LABEL: define{{.*}} void @_Z1fN1A1AENS_1BE(
void f(A::A a, A::B b) { }

struct C {
  struct D { };
};

// CHECK-LABEL: define{{.*}} void @_Z1fN1C1DERS_PS_S1_(
void f(C::D, C&, C*, C&) { }

template<typename T>
struct V {
  typedef int U;
};

template <typename T> void f1(typename V<T>::U, V<T>) { }

// CHECK: @_Z2f1IiEvN1VIT_E1UES2_
template void f1<int>(int, V<int>);

template <typename T> void f2(V<T>, typename V<T>::U) { }

// CHECK: @_Z2f2IiEv1VIT_ENS2_1UE
template void f2<int>(V<int>, int);

namespace NS {
template <typename T> struct S1 {};
template<typename T> void ft3(S1<T>, S1<char>) {  }

// CHECK: @_ZN2NS3ft3IiEEvNS_2S1IT_EENS1_IcEE
template void ft3<int>(S1<int>, S1<char>);
}

// PR5196
// CHECK: @_Z1fPKcS0_
void f(const char*, const char*) {}

namespace NS {
  class C;
}

namespace NS {
  // CHECK: @_ZN2NS1fERNS_1CE
  void f(C&) { } 
}

namespace Test1 {

struct A { };
struct B { };

// CHECK: @_ZN5Test11fEMNS_1BEFvvENS_1AES3_
void f(void (B::*)(), A, A) { }

// CHECK: @_ZN5Test11fEMNS_1BEFvvENS_1AES3_MS0_FvS3_EMS3_FvvE
void f(void (B::*)(), A, A, void (B::*)(A), void (A::*)()) { }

}

namespace ManglePrefix {
template <typename>
struct X {
  template <typename>
  struct Y {
    typedef int type;
    typedef int type2;
  };
};
template <typename T>
typename X<T>::template Y<T>::type f(typename X<T>::template Y<T>::type2) { return 0; }

// CHECK: @_ZN12ManglePrefix1fIiEENS_1XIT_E1YIS2_E4typeENS5_5type2E
template int f<int>(int);
}
