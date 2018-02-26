// RUN: %clang_cc1 -std=c++1y -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s

// Check that we keep the 'extern' when we instantiate the definition of this
// variable template specialization.
template<typename T> extern const int extern_redecl;
template<typename T> const int extern_redecl = 5;
template const int extern_redecl<int>;

// CHECK: @_Z13extern_redeclIiE = weak_odr constant

template<typename T> struct Outer {
  template<typename U> struct Inner {
    template<typename V> static int arr[];
  };
};
Outer<char[100]> outer_int;
int init_arr();
template<typename T> template<typename U> template<typename V> int Outer<T>::Inner<U>::arr[sizeof(T) + sizeof(U) + sizeof(V)] = { init_arr() };
int *p = Outer<char[100]>::Inner<char[20]>::arr<char[3]>;

namespace PR35456 {
// CHECK: @_ZN7PR354561nILi0EEE = linkonce_odr global i32 0
template<int> int n;
int *p = &n<0>;
}

// CHECK: @_ZN5OuterIA100_cE5InnerIA20_cE3arrIA3_cEE = linkonce_odr global [123 x i32] zeroinitializer
// CHECK: @_ZGVN5OuterIA100_cE5InnerIA20_cE3arrIA3_cEE = linkonce_odr global

// CHECK: call {{.*}}@_Z8init_arrv
