// RUN: %clang_cc1 -fvisibility=hidden -triple %itanium_abi_triple -emit-llvm -o - %s | FileCheck %s

// Verify that symbols are hidden.
// CHECK: @_ZN1CIiE5Inner6Inner26StaticE = weak_odr hidden global
// CHECK-LABEL: define weak_odr hidden {{.*}}void @_ZN1CIiE5Inner1fEv
// CHECK-LABEL: define weak_odr hidden {{.*}}void @_ZN1CIiE5Inner6Inner21gEv

template<typename T>
struct C {
  struct Inner {
    void f();
    struct Inner2 {
      void g();
      static int Static;
    };
  };
};

template<typename T> void C<T>::Inner::f() { }
template<typename T> void C<T>::Inner::Inner2::g() { }
template<typename T> int C<T>::Inner::Inner2::Static;

extern template struct C<int>;
template struct C<int>;
