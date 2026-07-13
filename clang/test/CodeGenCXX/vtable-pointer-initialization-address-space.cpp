// RUN: %clang_cc1 %s -triple=amdgcn-amd-amdhsa -std=c++11 -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple=spirv64-amd-amdhsa -std=c++11 -emit-llvm -o - | FileCheck %s --check-prefix=WITH-NONZERO-DEFAULT-AS

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
// CHECK: store ptr addrspace(1) getelementptr inbounds inrange(-16, 8) ({ [3 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTV1A, i32 0, i32 0, i32 2)
// CHECK: call void @_ZN5FieldC1Ev(
// CHECK: ret void
// WITH-NONZERO-DEFAULT-AS-LABEL: define{{.*}} void @_ZN1AC2Ev(ptr addrspace(4) {{[^,]*}} %this) unnamed_addr
A::A() { }

// CHECK-LABEL: define{{.*}} void @_ZN1AD2Ev(ptr {{[^,]*}} %this) unnamed_addr
// CHECK: store ptr addrspace(1) getelementptr inbounds inrange(-16, 8) ({ [3 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTV1A, i32 0, i32 0, i32 2)
// CHECK: call void @_ZN5FieldD1Ev(
// CHECK: call void @_ZN4BaseD2Ev(
// CHECK: ret void
// WITH-NONZERO-DEFAULT-AS-LABEL: define{{.*}} void @_ZN1AD2Ev(ptr addrspace(4) {{[^,]*}} %this) unnamed_addr
A::~A() { }

struct B : Base {
  virtual void f();

  Field field;
};

void f() { B b; }

// CHECK-LABEL: define linkonce_odr void @_ZN1BC1Ev(ptr {{[^,]*}} %this) unnamed_addr
// CHECK: call void @_ZN1BC2Ev(
// WITH-NONZERO-DEFAULT-AS-LABEL: define linkonce_odr{{.*}} void @_ZN1BC1Ev(ptr addrspace(4) {{[^,]*}} %this) unnamed_addr

// CHECK-LABEL: define linkonce_odr void @_ZN1BD1Ev(ptr {{[^,]*}} %this) unnamed_addr
// CHECK: call void @_ZN1BD2Ev(
// WITH-NONZERO-DEFAULT-AS-LABEL: define linkonce_odr{{.*}} void @_ZN1BD1Ev(ptr addrspace(4) {{[^,]*}} %this) unnamed_addr

// CHECK-LABEL: define linkonce_odr void @_ZN1BC2Ev(ptr {{[^,]*}} %this) unnamed_addr
// CHECK: call void @_ZN4BaseC2Ev(
// CHECK: store ptr addrspace(1) getelementptr inbounds inrange(-16, 8) ({ [3 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTV1B, i32 0, i32 0, i32 2)
// CHECK: call void @_ZN5FieldC1Ev
// CHECK: ret void
// WITH-NONZERO-DEFAULT-AS-LABEL: define linkonce_odr{{.*}} void @_ZN1BC2Ev(ptr addrspace(4) {{[^,]*}} %this) unnamed_addr

// CHECK-LABEL: define linkonce_odr void @_ZN1BD2Ev(ptr {{[^,]*}} %this) unnamed_addr
// CHECK: store ptr addrspace(1) getelementptr inbounds inrange(-16, 8) ({ [3 x ptr addrspace(1)] }, ptr addrspace(1) @_ZTV1B, i32 0, i32 0, i32 2)
// CHECK: call void @_ZN5FieldD1Ev(
// CHECK: call void @_ZN4BaseD2Ev(
// CHECK: ret void
// WITH-NONZERO-DEFAULT-AS-LABEL: define linkonce_odr{{.*}} void @_ZN1BD2Ev(ptr addrspace(4) {{[^,]*}} %this) unnamed_addr
