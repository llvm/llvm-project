// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

struct Field {
  Field();
  ~Field();
};

struct Base {
  Base();
  ~Base();
};

struct A : Base {
  A();
  ~A();

  virtual void f();
  
  Field field;
};

// CHECK-LABEL: define{{.*}} void @_ZN1AC2Ev(ptr {{[^,]*}} %this) unnamed_addr
// CHECK: call void @_ZN4BaseC2Ev(
// CHECK: store ptr getelementptr inbounds ({ [3 x ptr] }, ptr @_ZTV1A, i32 0, inrange i32 0, i32 2)
// CHECK: call void @_ZN5FieldC1Ev(
// CHECK: ret void
A::A() { }

// CHECK-LABEL: define{{.*}} void @_ZN1AD2Ev(ptr {{[^,]*}} %this) unnamed_addr
// CHECK: store ptr getelementptr inbounds ({ [3 x ptr] }, ptr @_ZTV1A, i32 0, inrange i32 0, i32 2)
// CHECK: call void @_ZN5FieldD1Ev(
// CHECK: call void @_ZN4BaseD2Ev(
// CHECK: ret void
A::~A() { } 

struct B : Base {
  virtual void f();
  
  Field field;
};

void f() { B b; }

// CHECK-LABEL: define linkonce_odr void @_ZN1BC1Ev(ptr {{[^,]*}} %this) unnamed_addr
// CHECK: call void @_ZN1BC2Ev(

// CHECK-LABEL: define linkonce_odr void @_ZN1BD1Ev(ptr {{[^,]*}} %this) unnamed_addr
// CHECK: call void @_ZN1BD2Ev(

// CHECK-LABEL: define linkonce_odr void @_ZN1BC2Ev(ptr {{[^,]*}} %this) unnamed_addr
// CHECK: call void @_ZN4BaseC2Ev(
// CHECK: store ptr getelementptr inbounds ({ [3 x ptr] }, ptr @_ZTV1B, i32 0, inrange i32 0, i32 2)
// CHECK: call void @_ZN5FieldC1Ev
// CHECK: ret void

// CHECK-LABEL: define linkonce_odr void @_ZN1BD2Ev(ptr {{[^,]*}} %this) unnamed_addr
// CHECK: store ptr getelementptr inbounds ({ [3 x ptr] }, ptr @_ZTV1B, i32 0, inrange i32 0, i32 2)
// CHECK: call void @_ZN5FieldD1Ev(
// CHECK: call void @_ZN4BaseD2Ev(
// CHECK: ret void
