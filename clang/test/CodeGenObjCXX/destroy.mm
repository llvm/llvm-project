// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-arc -disable-llvm-passes -o - %s | FileCheck %s
// rdar://18249673

@class MyObject;
struct base {
  ~base() = default;
};
struct derived : public base {
  MyObject *myobject;
};

void test1() {
  derived d1;
}
// CHECK-LABEL: define void @_Z5test1v()
// CHECK: call void @_ZN7derivedC1Ev
// CHECK: call void @_ZN7derivedD1Ev

void test2() {
  derived *d2 = new derived;
  delete d2;
}
// CHECK-LABEL: define void @_Z5test2v()
// CHECK:   call void @_ZN7derivedC1Ev
// CHECK:   call void @_ZN7derivedD1Ev

template <typename T>
struct tderived : public base {
  MyObject *myobject;
};
void test3() {
  tderived<int> d1;
}
// CHECK-LABEL: define void @_Z5test3v()
// CHECK: call void @_ZN8tderivedIiEC1Ev
// CHECK: call void @_ZN8tderivedIiED1Ev

void test4() {
  tderived<int> *d2 = new tderived<int>;
  delete d2;
}
// CHECK-LABEL: define void @_Z5test4v()
// CHECK: call void @_ZN8tderivedIiEC1Ev
// CHECK: call void @_ZN8tderivedIiED1Ev

// CHECK-LABEL: define linkonce_odr void @_ZN7derivedD2Ev
// CHECK: call void @objc_storeStrong(i8** {{.*}}, i8* null)

// CHECK-LABEL: define linkonce_odr void @_ZN8tderivedIiED2Ev
// CHECK: call void @objc_storeStrong(i8** {{.*}}, i8* null)
