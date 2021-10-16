// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -emit-llvm -w -o - | FileCheck %s

struct S {
  enum { FOO = 42 };
  enum { BAR = 42 };
};

template <typename T> struct X {
  T value;
  X(T t) : value(t) {}
  int f() { return value; }
};

template <typename T> int f(T t) {
  X<T> x(t);
  return x.f();
}

void test() {
  // Look for two instantiations, one for FOO's
  // type and one for BAR's.
  // CHECK-LABEL: define linkonce_odr noundef i32 @_Z1fIN1SUt_EEiT_(i32 noundef %t)
  (void)f(S::FOO);
  // CHECK-LABEL: define linkonce_odr noundef i32 @_Z1fIN1SUt0_EEiT_(i32 noundef %t)
  (void)f(S::BAR);

  // Now check for the class template instantiations.
  //
  // BAR's instantiation of X:
  // CHECK-LABEL: define linkonce_odr noundef i32 @_ZN1XIN1SUt_EE1fEv(%struct.X* {{[^,]*}} %this)
  // CHECK-LABEL: define linkonce_odr void @_ZN1XIN1SUt_EEC2ES1_(%struct.X* {{[^,]*}} %this, i32 noundef %t) unnamed_addr
  //
  // FOO's instantiation of X:
  // CHECK-LABEL: define linkonce_odr noundef i32 @_ZN1XIN1SUt0_EE1fEv(%struct.X.0* {{[^,]*}} %this)
  // CHECK-LABEL: define linkonce_odr void @_ZN1XIN1SUt0_EEC2ES1_(%struct.X.0* {{[^,]*}} %this, i32 noundef %t) unnamed_addr
}
