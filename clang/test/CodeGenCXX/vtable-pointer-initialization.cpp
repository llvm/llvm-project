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

// CHECK: define void @_ZN1AC2Ev(%struct.A* %this) unnamed_addr
// CHECK: call void @_ZN4BaseC2Ev(
// CHECK: store i8** getelementptr inbounds ([3 x i8*]* @_ZTV1A, i64 0, i64 2)
// CHECK: call void @_ZN5FieldC1Ev(
// CHECK: ret void
A::A() { }

// CHECK: define void @_ZN1AD2Ev(%struct.A* %this) unnamed_addr
// CHECK: store i8** getelementptr inbounds ([3 x i8*]* @_ZTV1A, i64 0, i64 2)
// CHECK: call void @_ZN5FieldD1Ev(
// CHECK: call void @_ZN4BaseD2Ev(
// CHECK: ret void
A::~A() { } 

struct B : Base {
  virtual void f();
  
  Field field;
};

void f() { B b; }

// CHECK: define linkonce_odr void @_ZN1BC1Ev(%struct.B* %this) unnamed_addr
// CHECK: call void @_ZN1BC2Ev(

// CHECK: define linkonce_odr void @_ZN1BD1Ev(%struct.B* %this) unnamed_addr
// CHECK: store i8** getelementptr inbounds ([3 x i8*]* @_ZTV1B, i64 0, i64 2)
// CHECK: call void @_ZN5FieldD1Ev(
// CHECK: call void @_ZN4BaseD2Ev(
// CHECK: ret void

// CHECK: define linkonce_odr void @_ZN1BC2Ev(%struct.B* %this) unnamed_addr
// CHECK: call void @_ZN4BaseC2Ev(
// CHECK: store i8** getelementptr inbounds ([3 x i8*]* @_ZTV1B, i64 0, i64 2)
// CHECK: call void @_ZN5FieldC1Ev
// CHECK: ret void
