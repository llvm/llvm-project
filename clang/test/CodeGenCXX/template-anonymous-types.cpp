// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -emit-llvm -o - | FileCheck %s

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
  // Look for two instantiations, entirely internal to this TU, one for FOO's
  // type and one for BAR's.
  // CHECK: define internal i32 @"_Z1fIN1S3$_0EEiT_"(i32 %t)
  (void)f(S::FOO);
  // CHECK: define internal i32 @"_Z1fIN1S3$_1EEiT_"(i32 %t)
  (void)f(S::BAR);

  // Now check for the class template instantiations. Annoyingly, they are in
  // reverse order.
  //
  // BAR's instantiation of X:
  // CHECK: define internal i32 @"_ZN1XIN1S3$_1EE1fEv"(%struct.X* %this)
  // CHECK: define internal void @"_ZN1XIN1S3$_1EEC2ES1_"(%struct.X* %this, i32 %t) unnamed_addr
  //
  // FOO's instantiation of X:
  // CHECK: define internal i32 @"_ZN1XIN1S3$_0EE1fEv"(%struct.X.0* %this)
  // CHECK: define internal void @"_ZN1XIN1S3$_0EEC2ES1_"(%struct.X.0* %this, i32 %t) unnamed_addr
}
