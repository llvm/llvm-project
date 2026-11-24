// RUN: %clang_cc1 %s -triple=amdgpu-amd-amdhsa -std=c++11 -emit-llvm -o - | FileCheck %s
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

// CHECK: %struct.B = type <{ ptr addrspace(1), [8 x i8] }>
// CHECK: %struct.B.base = type <{ ptr addrspace(1), i8 }>
// CHECK: %struct.K = type { ptr addrspace(1) }
// WITH-NONZERO-DEFAULT-AS: %struct.B = type <{ ptr addrspace(1), [8 x i8] }>
// WITH-NONZERO-DEFAULT-AS: %struct.B.base = type <{ ptr addrspace(1), i8 }>
// WITH-NONZERO-DEFAULT-AS: %struct.K = type { ptr addrspace(1) }

// CHECK-LABEL: define{{.*}} void @_ZN1AC2Ev(ptr {{[^,]*}} %this) unnamed_addr
// CHECK: call void @_ZN4BaseC2Ev(
// CHECK: store ptr addrspace(1) getelementptr inbounds inrange(-16, 8) (i8, ptr addrspace(1) @_ZTV1A, i64 16)
// CHECK: call void @_ZN5FieldC1Ev(
// CHECK: ret void
// WITH-NONZERO-DEFAULT-AS-LABEL: define{{.*}} void @_ZN1AC2Ev(ptr addrspace(4) {{[^,]*}} %this) unnamed_addr
A::A() { }

// CHECK-LABEL: define{{.*}} void @_ZN1AD2Ev(ptr {{[^,]*}} %this) unnamed_addr
// CHECK: store ptr addrspace(1) getelementptr inbounds inrange(-16, 8) (i8, ptr addrspace(1) @_ZTV1A, i64 16)
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
// CHECK: store ptr addrspace(1) getelementptr inbounds inrange(-16, 8) (i8, ptr addrspace(1) @_ZTV1B, i64 16)
// CHECK: call void @_ZN5FieldC1Ev
// CHECK: ret void
// WITH-NONZERO-DEFAULT-AS-LABEL: define linkonce_odr{{.*}} void @_ZN1BC2Ev(ptr addrspace(4) {{[^,]*}} %this) unnamed_addr

// CHECK-LABEL: define linkonce_odr void @_ZN1BD2Ev(ptr {{[^,]*}} %this) unnamed_addr
// CHECK: store ptr addrspace(1) getelementptr inbounds inrange(-16, 8) (i8, ptr addrspace(1) @_ZTV1B, i64 16)
// CHECK: call void @_ZN5FieldD1Ev(
// CHECK: call void @_ZN4BaseD2Ev(
// CHECK: ret void
// WITH-NONZERO-DEFAULT-AS-LABEL: define linkonce_odr{{.*}} void @_ZN1BD2Ev(ptr addrspace(4) {{[^,]*}} %this) unnamed_addr

struct K {
  virtual void g();
};

struct C : B, K {
  double x;
};

void g() { C c; }
