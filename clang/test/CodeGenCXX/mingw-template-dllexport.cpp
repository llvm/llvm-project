// RUN: %clang_cc1 -emit-llvm -triple i686-mingw32 %s -o - | FileCheck %s

#define JOIN2(x, y) x##y
#define JOIN(x, y) JOIN2(x, y)
#define UNIQ(name) JOIN(name, __LINE__)
#define USEMEMFUNC(class, func) void (class::*UNIQ(use)())() { return &class::func; }

template <class T>
class c {
  void f() {}
};

template class __declspec(dllexport) c<int>;

// CHECK: define {{.*}} dllexport {{.*}} @_ZN1cIiE1fEv

extern template class __declspec(dllexport) c<char>;
template class c<char>;

// CHECK: define {{.*}} dllexport {{.*}} @_ZN1cIcE1fEv

extern template class c<double>;
template class __declspec(dllexport) c<double>;

// CHECK-NOT: define {{.*}} dllexport {{.*}} @_ZN1cIdE1fEv

template <class T>
struct outer {
  void f();
  struct inner {
    void f();
  };
};

template <class T> void outer<T>::f() {}
template <class T> void outer<T>::inner::f() {}

template class __declspec(dllexport) outer<int>;

// CHECK: define {{.*}} dllexport {{.*}} @_ZN5outerIiE1fEv
// CHECK-NOT: define {{.*}} dllexport {{.*}} @_ZN5outerIiE5inner1fEv

extern template class __declspec(dllimport) outer<char>;
USEMEMFUNC(outer<char>, f)
USEMEMFUNC(outer<char>::inner, f)

// CHECK: declare dllimport {{.*}} @_ZN5outerIcE1fEv
// CHECK: define {{.*}} @_ZN5outerIcE5inner1fEv
