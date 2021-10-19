// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s | FileCheck %s 

template<class T> void f(T) { /* ... */ }
template<class T> inline void g(T) { /* ... */ }

// CHECK: define{{.*}} void @_Z1gIiEvT_
template<> void g<>(int) { /* ... */ }

template<class T>
struct X {
  void f() { }
  void g();
  void h();
};

template<class T>
void X<T>::g() {
}

template<class T>
inline void X<T>::h() {
}

// CHECK: define{{.*}} void @_ZN1XIiE1fEv
template<> void X<int>::f() { }

// CHECK: define{{.*}} void @_ZN1XIiE1hEv
template<> void X<int>::h() { }

// CHECK: define linkonce_odr void @_Z1fIiEvT_
template<> inline void f<>(int) { /* ... */ } 

// CHECK: define linkonce_odr void @_ZN1XIiE1gEv
template<> inline void X<int>::g() { }

void test(X<int> xi) {
  f(17);
  g(17);
  xi.f();
  xi.g();
  xi.h();
}
