// RUN: %clang_cc1 -fms-compatibility -triple x86_64-windows-msvc %s -emit-llvm -o - | FileCheck %s

// Make sure we choose the *direct* base path when doing these conversions.

struct A { int a; };
struct B : A { int b; };

struct C : A, B { };
extern "C" A *a_from_c(C *p) { return p; }
// CHECK-LABEL: define dso_local ptr @a_from_c(ptr noundef %{{.*}})
// CHECK: [[P_ADDR:%.*]] = alloca ptr
// CHECK-NEXT: store ptr [[P:%.*]], ptr [[P_ADDR]]
// CHECK-NEXT: [[RET:%.*]] = load ptr, ptr [[P_ADDR]]
// CHECK-NEXT: ret ptr [[RET]]

struct D : B, A { };
extern "C" A *a_from_d(D *p) { return p; }
// CHECK-LABEL: define dso_local ptr @a_from_d(ptr noundef %{{.*}})
// CHECK: [[P_ADDR:%.*]] = alloca ptr
// CHECK-NEXT: store ptr [[P:%.*]], ptr [[P_ADDR]]
// CHECK-NEXT: [[P_RELOAD:%.*]] = load ptr, ptr [[P_ADDR]]
// CHECK-NEXT: [[CMP:%.*]] = icmp eq ptr [[P_RELOAD]], null
// CHECK: [[ADD_PTR:%.*]] = getelementptr inbounds i8, ptr [[P_RELOAD]], i64 8
// CHECK: [[RET:%.*]] = phi ptr [ [[ADD_PTR]], {{.*}} ], [ null, %entry ]
// CHECK-NEXT: ret ptr [[RET]]
