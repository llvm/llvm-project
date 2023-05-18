// RUN: %clang_cc1 -emit-llvm -triple x86_64-apple-darwin10 -o - %s | FileCheck %s
template<typename T>
struct X {
  static T member1;
  static T member2;
  static T member3;
};

template<typename T>
T X<T>::member1;

template<typename T>
T X<T>::member2 = 17;

// CHECK: @_ZN1XIiE7member1E = weak_odr global i32 0
template int X<int>::member1;

// CHECK: @_ZN1XIiE7member2E = weak_odr global i32 17
template int X<int>::member2;

// For implicit instantiation of 
long& get(bool Cond1, bool Cond2) {
  // CHECK: @_ZN1XIlE7member1E = linkonce_odr global i64 0
  // CHECK: @_ZN1XIlE7member2E = linkonce_odr global i64 17
  // CHECK: @_ZN1XIlE7member3E = external global i64
  return Cond1? X<long>::member1 
       : Cond2? X<long>::member2
              : X<long>::member3;
}
