// RUN: %clang_cc1 -triple=x86_64-apple-darwin -std=c++11 -emit-llvm -o - %s | FileCheck %s

class B {
public:
  [[clang::disable_tail_calls]] virtual int m1() { return 1; }
  virtual int m2() { return 2; }
  int m3() { return 3; }
  [[clang::disable_tail_calls]] int m4();
};

class D : public B {
public:
  int m1() override { return 11; }
  [[clang::disable_tail_calls]] int m2() override { return 22; }
};

int foo1() {
  B *b = new B;
  D *d = new D;
  int t = 0;
  t += b->m1() + b->m2() + b->m3() + b->m4();
  t += d->m1() + d->m2();
  return t;
}

// CHECK: define linkonce_odr i32 @_ZN1B2m3Ev(%class.B* {{[^,]*}} %this) [[ATTRFALSE:#[0-9]+]]
// CHECK: declare i32 @_ZN1B2m4Ev(%class.B* {{[^,]*}}) [[ATTRTRUE0:#[0-9]+]]
// CHECK: define linkonce_odr i32 @_ZN1B2m1Ev(%class.B* {{[^,]*}} %this) unnamed_addr [[ATTRTRUE1:#[0-9]+]]
// CHECK: define linkonce_odr i32 @_ZN1B2m2Ev(%class.B* {{[^,]*}} %this) unnamed_addr [[ATTRFALSE:#[0-9]+]]
// CHECK: define linkonce_odr i32 @_ZN1D2m1Ev(%class.D* {{[^,]*}} %this) unnamed_addr [[ATTRFALSE:#[0-9]+]]
// CHECK: define linkonce_odr i32 @_ZN1D2m2Ev(%class.D* {{[^,]*}} %this) unnamed_addr [[ATTRTRUE1:#[0-9]+]]

// CHECK: attributes [[ATTRFALSE]] = { {{.*}}"disable-tail-calls"="false"{{.*}} }
// CHECK: attributes [[ATTRTRUE0]] = { {{.*}}"disable-tail-calls"="true"{{.*}} }
// CHECK: attributes [[ATTRTRUE1]] = { {{.*}}"disable-tail-calls"="true"{{.*}} }
