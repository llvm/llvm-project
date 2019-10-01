// RUN: %clang_cc1 -O1 -fno-experimental-new-pass-manager -triple %itanium_abi_triple -emit-llvm -o - -fvisibility hidden %s | FileCheck %s

template<typename T>
struct X {
  void f();
  void g() { }
};

template<typename T> void X<T>::f() { }

extern template struct X<int>;
template struct X<int>;
extern template struct X<char>;

// <rdar://problem/8109763>
void test_X(X<int> xi, X<char> xc) {
  // CHECK-LABEL: define weak_odr hidden {{.*}}void @_ZN1XIiE1fEv
  xi.f();
  // CHECK-LABEL: define weak_odr hidden {{.*}}void @_ZN1XIiE1gEv
  xi.g();
  // CHECK: declare {{.*}}void @_ZN1XIcE1fEv
  xc.f();
  // CHECK-LABEL: define available_externally {{.*}}void @_ZN1XIcE1gEv
  xc.g();
}

