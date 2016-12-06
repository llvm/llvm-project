// RUN: %clang_cc1 -emit-llvm -triple i686-windows-itanium -fdeclspec %s -o - | FileCheck %s

struct __declspec(dllexport) s {
  void f() {}
};

// CHECK: define {{.*}} dllexport {{.*}} @_ZN1saSERKS_
// CHECK: define {{.*}} dllexport {{.*}} @_ZN1s1fEv

template <class T>
class c {
  void f() {}
};

template class __declspec(dllexport) c<int>;

// CHECK: define {{.*}} dllexport {{.*}} @_ZN1cIiEaSERKS0_
// CHECK: define {{.*}} dllexport {{.*}} @_ZN1cIiE1fEv

extern template class c<char>;
template class __declspec(dllexport) c<char>;

// CHECK: define {{.*}} dllexport {{.*}} @_ZN1cIcEaSERKS0_
// CHECK: define {{.*}} dllexport {{.*}} @_ZN1cIcE1fEv

c<double> g;
template class __declspec(dllexport) c<double>;

// CHECK: define {{.*}} dllexport {{.*}} @_ZN1cIdEaSERKS0_
// CHECK: define {{.*}} dllexport {{.*}} @_ZN1cIdE1fEv
