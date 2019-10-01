// RUN: %clang_cc1 -O0 -triple amdgcn -emit-llvm %s -o - | FileCheck %s

class A {
public:
  int x;
  A():x(0) {}
  ~A() {}
};

class B {
int x[100];
};

A g_a;
B g_b;

void func_with_ref_arg(A &a);
void func_with_ref_arg(B &b);

// CHECK-LABEL: define void @_Z22func_with_indirect_arg1A(%class.A addrspace(5)* %a)
// CHECK:  %p = alloca %class.A*, align 8, addrspace(5)
// CHECK:  %[[r1:.+]] = addrspacecast %class.A* addrspace(5)* %p to %class.A**
// CHECK:  %[[r0:.+]] = addrspacecast %class.A addrspace(5)* %a to %class.A*
// CHECK:  store %class.A* %[[r0]], %class.A** %[[r1]], align 8
void func_with_indirect_arg(A a) {
  A *p = &a;
}

// CHECK-LABEL: define void @_Z22test_indirect_arg_autov()
// CHECK:  %a = alloca %class.A, align 4, addrspace(5)
// CHECK:  %[[r0:.+]] = addrspacecast %class.A addrspace(5)* %a to %class.A*
// CHECK:  %agg.tmp = alloca %class.A, align 4, addrspace(5)
// CHECK:  %[[r1:.+]] = addrspacecast %class.A addrspace(5)* %agg.tmp to %class.A*
// CHECK:  call void @_ZN1AC1Ev(%class.A* %[[r0]])
// CHECK:  call void @llvm.memcpy.p0i8.p0i8.i64
// CHECK:  %[[r4:.+]] = addrspacecast %class.A* %[[r1]] to %class.A addrspace(5)*
// CHECK:  call void @_Z22func_with_indirect_arg1A(%class.A addrspace(5)* %[[r4]])
// CHECK:  call void @_ZN1AD1Ev(%class.A* %[[r1]])
// CHECK:  call void @_Z17func_with_ref_argR1A(%class.A* dereferenceable(4) %[[r0]])
// CHECK:  call void @_ZN1AD1Ev(%class.A* %[[r0]])
void test_indirect_arg_auto() {
  A a;
  func_with_indirect_arg(a);
  func_with_ref_arg(a);
}

// CHECK: define void @_Z24test_indirect_arg_globalv()
// CHECK:  %agg.tmp = alloca %class.A, align 4, addrspace(5)
// CHECK:  %[[r0:.+]] = addrspacecast %class.A addrspace(5)* %agg.tmp to %class.A*
// CHECK:  call void @llvm.memcpy.p0i8.p0i8.i64
// CHECK:  %[[r2:.+]] = addrspacecast %class.A* %[[r0]] to %class.A addrspace(5)*
// CHECK:  call void @_Z22func_with_indirect_arg1A(%class.A addrspace(5)* %[[r2]])
// CHECK:  call void @_ZN1AD1Ev(%class.A* %[[r0]])
// CHECK:  call void @_Z17func_with_ref_argR1A(%class.A* dereferenceable(4) addrspacecast (%class.A addrspace(1)* @g_a to %class.A*))
void test_indirect_arg_global() {
  func_with_indirect_arg(g_a);
  func_with_ref_arg(g_a);
}

// CHECK-LABEL: define void @_Z19func_with_byval_arg1B(%class.B addrspace(5)* byval(%class.B) align 4 %b)
// CHECK:  %p = alloca %class.B*, align 8, addrspace(5)
// CHECK:  %[[r1:.+]] = addrspacecast %class.B* addrspace(5)* %p to %class.B**
// CHECK:  %[[r0:.+]] = addrspacecast %class.B addrspace(5)* %b to %class.B*
// CHECK:  store %class.B* %[[r0]], %class.B** %[[r1]], align 8
void func_with_byval_arg(B b) {
  B *p = &b;
}

// CHECK-LABEL: define void @_Z19test_byval_arg_autov()
// CHECK:  %b = alloca %class.B, align 4, addrspace(5)
// CHECK:  %[[r0:.+]] = addrspacecast %class.B addrspace(5)* %b to %class.B*
// CHECK:  %agg.tmp = alloca %class.B, align 4, addrspace(5)
// CHECK:  %[[r1:.+]] = addrspacecast %class.B addrspace(5)* %agg.tmp to %class.B*
// CHECK:  call void @llvm.memcpy.p0i8.p0i8.i64
// CHECK:  %[[r4:.+]] = addrspacecast %class.B* %[[r1]] to %class.B addrspace(5)*
// CHECK:  call void @_Z19func_with_byval_arg1B(%class.B addrspace(5)* byval(%class.B) align 4 %[[r4]])
// CHECK:  call void @_Z17func_with_ref_argR1B(%class.B* dereferenceable(400) %[[r0]])
void test_byval_arg_auto() {
  B b;
  func_with_byval_arg(b);
  func_with_ref_arg(b);
}

// CHECK-LABEL: define void @_Z21test_byval_arg_globalv()
// CHECK:  %agg.tmp = alloca %class.B, align 4, addrspace(5)
// CHECK:  %[[r0:.+]] = addrspacecast %class.B addrspace(5)* %agg.tmp to %class.B*
// CHECK:  call void @llvm.memcpy.p0i8.p0i8.i64
// CHECK:  %[[r2:.+]] = addrspacecast %class.B* %[[r0]] to %class.B addrspace(5)*
// CHECK:  call void @_Z19func_with_byval_arg1B(%class.B addrspace(5)* byval(%class.B) align 4 %[[r2]])
// CHECK:  call void @_Z17func_with_ref_argR1B(%class.B* dereferenceable(400) addrspacecast (%class.B addrspace(1)* @g_b to %class.B*))
void test_byval_arg_global() {
  func_with_byval_arg(g_b);
  func_with_ref_arg(g_b);
}
