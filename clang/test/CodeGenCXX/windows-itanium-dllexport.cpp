// RUN: %clang_cc1 -emit-llvm -triple i686-windows-itanium -fdeclspec %s -o - | FileCheck %s --check-prefixes=CHECK,WI
// RUN: %clang_cc1 -emit-llvm -triple x86_64-scei-ps4 -fdeclspec %s -o - | FileCheck %s --check-prefixes=CHECK,PS4
// RUN: %clang_cc1 -emit-llvm -triple x86_64-sie-ps5  -fdeclspec %s -o - | FileCheck %s --check-prefixes=CHECK,PS4

#define JOIN2(x, y) x##y
#define JOIN(x, y) JOIN2(x, y)
#define UNIQ(name) JOIN(name, __LINE__)
#define USEMEMFUNC(class, func) void (class::*UNIQ(use)())() { return &class::func; }

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

// WI: define {{.*}} dllexport {{.*}} @_ZN1cIcEaSERKS0_
// WI: define {{.*}} dllexport {{.*}} @_ZN1cIcE1fEv
// PS4-NOT: @_ZN1cIcEaSERKS0_
// PS4: define weak_odr void @_ZN1cIcE1fEv

c<double> g;
template class __declspec(dllexport) c<double>;

// WI: define {{.*}} dllexport {{.*}} @_ZN1cIdEaSERKS0_
// WI: define {{.*}} dllexport {{.*}} @_ZN1cIdE1fEv
// PS4-NOT: @_ZN1cIdEaSERKS0_
// PS4: define weak_odr void @_ZN1cIdE1fEv

template <class T>
struct outer {
  void f() {}
  struct inner {
    void f() {}
  };
};

template class __declspec(dllexport) outer<int>;

// CHECK: define {{.*}} dllexport {{.*}} @_ZN5outerIiE1fEv
// CHECK-NOT: define {{.*}} dllexport {{.*}} @_ZN5outerIiE5inner1fEv

extern template class __declspec(dllimport) outer<char>;
USEMEMFUNC(outer<char>, f)
USEMEMFUNC(outer<char>::inner, f)

// CHECK-DAG: declare dllimport {{.*}} @_ZN5outerIcE1fEv
// WI-DAG: define {{.*}} @_ZN5outerIcE5inner1fEv
// PS4-DAG: declare {{.*}} @_ZN5outerIcE5inner1fEv
