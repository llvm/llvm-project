// RUN: %clang_cc1 -emit-llvm -triple i686-mingw32 %s -o - | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -triple x86_64-w64-mingw32 %s -o - | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -triple i686-pc-cygwin %s -o - | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -triple x86_64-pc-cygwin %s -o - | FileCheck %s

#define JOIN2(x, y) x##y
#define JOIN(x, y) JOIN2(x, y)
#define UNIQ(name) JOIN(name, __LINE__)
#define USEMEMFUNC(class, func) auto UNIQ(use) = &class::func;

template <class T>
class c {
  // MinGW-GCC does not apply 'dllexport' to inline member function in dll-exported template but clang does from long ago.
  void f() {}
  void g();
  inline static int u = 0;
  static int v;
};
template <class T> void c<T>::g() {}
template <class T> int c<T>::v = 0;

// #1
template class __declspec(dllexport) c<int>;

// #2
extern template class __declspec(dllexport) c<char>;
template class c<char>;

// #3
extern template class c<double>;
template class __declspec(dllexport) c<double>; // expected-warning {{ 'dllexport' attribute ignored on explicit instantiation definition }}


template <class T>
struct outer {
  void f() {}
  void g();
  inline static int u = 0;
  static int v;
  // MinGW-GCC and Clang does not apply 'dllexport' to inner type and its sub-elements in template class.
  struct inner {
    void f() {}
    void g();
    inline static int u = 0;
    static int v;
  };
};

template <class T> void outer<T>::g() {}
template <class T> void outer<T>::inner::g() {}
template <class T> int outer<T>::v = 0;
template <class T> int outer<T>::inner::v = 0;

// #4
template struct __declspec(dllexport) outer<int>;

// #5
extern template struct __declspec(dllimport) outer<char>;
USEMEMFUNC(outer<char>, f)
USEMEMFUNC(outer<char>, g)
USEMEMFUNC(outer<char>, u)
USEMEMFUNC(outer<char>, v)
USEMEMFUNC(outer<char>::inner, f)
USEMEMFUNC(outer<char>::inner, g)
USEMEMFUNC(outer<char>::inner, u)
USEMEMFUNC(outer<char>::inner, v)


// #1 variables
// CHECK: @_ZN1cIiE1uE = {{.*}} dllexport {{.*}}
// CHECK: @_ZN1cIiE1vE = {{.*}} dllexport {{.*}}

// #2 variables
// CHECK: @_ZN1cIcE1uE = {{.*}} dllexport {{.*}}
// CHECK: @_ZN1cIcE1vE = {{.*}} dllexport {{.*}}

// #3 variables
// CHECK: @_ZN1cIdE1uE = {{.*}}
// CHECK-NOT: @_ZN1cIcE1uE = {{.*}} dllexport {{.*}}
// CHECK: @_ZN1cIdE1vE = {{.*}}
// CHECK-NOT: @_ZN1cIcE1vE = {{.*}} dllexport {{.*}}

// #4 variables
// CHECK: @_ZN5outerIiE1uE = {{.*}} dllexport {{.*}}
// CHECK: @_ZN5outerIiE1vE = {{.*}} dllexport {{.*}}
// CHECK: @_ZN5outerIiE5inner1uE = {{.*}}
// CHECK-NOT: @_ZN5outerIiE5inner1uE = {{.*}} dllexport {{.*}}
// CHECK: @_ZN5outerIiE5inner1vE = {{.*}}
// CHECK-NOT: @_ZN5outerIiE5inner1vE = {{.*}} dllexport {{.*}}

// #5 variables
// CHECK: @_ZN5outerIcE1uE = external dllimport {{.*}}
// CHECK: @_ZN5outerIcE1vE = external dllimport {{.*}}
// CHECK-NOT: @_ZN5outerIcE5inner1uE = dllimport {{.*}}
// CHECK-NOT: @_ZN5outerIcE5inner1vE = dllimport {{.*}}
// CHECK: @_ZN5outerIcE5inner1uE = external {{.*}}
// CHECK: @_ZN5outerIcE5inner1vE = external {{.*}}


// #1 functions
// CHECK: define {{.*}} dllexport {{.*}} @_ZN1cIiE1fEv
// CHECK: define {{.*}} dllexport {{.*}} @_ZN1cIiE1gEv

// #2 functions
// CHECK: define {{.*}} dllexport {{.*}} @_ZN1cIcE1fEv
// CHECK: define {{.*}} dllexport {{.*}} @_ZN1cIcE1gEv

// #3 functions
// CHECK-NOT: define {{.*}} dllexport {{.*}} @_ZN1cIdE1fEv
// CHECK-NOT: define {{.*}} dllexport {{.*}} @_ZN1cIdE1gEv

// #4 functions
// CHECK: define {{.*}} dllexport {{.*}} @_ZN5outerIiE1fEv
// CHECK: define {{.*}} dllexport {{.*}} @_ZN5outerIiE1gEv
// CHECK-NOT: define {{.*}} dllexport {{.*}} @_ZN5outerIiE5inner1fEv
// CHECK-NOT: define {{.*}} dllexport {{.*}} @_ZN5outerIiE5inner1gEv

// #5 functions
// CHECK: declare dllimport {{.*}} @_ZN5outerIcE1fEv
// CHECK: declare dllimport {{.*}} @_ZN5outerIcE1gEv
// CHECK-NOT: declare dllimport {{.*}} @_ZN5outerIcE5inner1fEv
// CHECK-NOT: declare dllimport {{.*}} @_ZN5outerIcE5inner1gEv
// CHECK-NOT: define {{.*}} @_ZN5outerIcE1fEv
// CHECK-NOT: define {{.*}} @_ZN5outerIcE5inner1fEv
// CHECK-NOT: define {{.*}} @_ZN5outerIcE1gEv
// CHECK-NOT: define {{.*}} @_ZN5outerIcE5inner1gEv
