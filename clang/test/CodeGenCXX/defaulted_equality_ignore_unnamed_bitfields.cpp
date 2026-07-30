// RUN: %clang_cc1 -std=c++20 %s -triple x86_64-linux -emit-llvm -o - | FileCheck %s

// GH61417
// Check that we don't attempt to compare the unnamed bitfields
struct A {
  unsigned x : 1;
  unsigned   : 1;

  friend bool operator==(A, A);
};


struct B {
  unsigned x : 1;
  unsigned   : 31;

  friend bool operator==(B, B);
};

bool operator==(A, A) = default;
// CHECK: define{{.*}} @_Zeq1AS_
// CHECK: %[[LHS:.+]] = alloca %struct.A, align 4
// CHECK: %[[RHS:.+]] = alloca %struct.A, align 4
// CHECK: %[[LHS_LOAD:.+]] = load i8, ptr %[[LHS]], align 4
// CHECK: %[[LHS_CLEAR:.+]] = and i8 %[[LHS_LOAD]], 1
// CHECK: %[[LHS_CAST:.+]] = zext i8 %[[LHS_CLEAR]] to i32

// CHECK: %[[RHS_LOAD:.+]] = load i8, ptr %[[RHS]]
// CHECK: %[[RHS_CLEAR:.+]] = and i8 %[[RHS_LOAD]], 1
// CHECK: %[[RHS_CAST:.+]] = zext i8 %[[RHS_CLEAR]] to i32
// CHECK: %[[CMP:.*]] = icmp eq i32 %[[LHS_CAST]], %[[RHS_CAST]]
// CHECK: ret i1 %[[CMP]]

bool operator==(B, B) = default;
// CHECK: define{{.*}} @_Zeq1BS_
// CHECK: %[[LHS_B:.+]] = alloca %struct.B, align 4
// CHECK: %[[RHS_B:.+]] = alloca %struct.B, align 4
// CHECK: %[[LHS_LOAD_B:.+]] = load i32, ptr %[[LHS_B]], align 4
// CHECK: %[[LHS_CLEAR_B:.+]] = and i32 %[[LHS_LOAD_B]], 1

// CHECK: %[[RHS_LOAD_B:.+]] = load i32, ptr %[[RHS_B]]
// CHECK: %[[RHS_CLEAR_B:.+]] = and i32 %[[RHS_LOAD_B]], 1
// CHECK: %[[CMP_B:.*]] = icmp eq i32 %[[LHS_CLEAR_B]], %[[RHS_CLEAR_B]]
// CHECK: ret i1 %[[CMP_B]]
