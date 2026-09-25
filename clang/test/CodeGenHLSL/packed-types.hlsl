// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple dxil-pc-shadermodel6.6-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s

// CHECK: define {{.*}} void @_Z1fu14int8_t4_packedu15uint8_t4_packed(i32 %{{.*}}, i32 %{{.*}})
void f(int8_t4_packed s_arg, uint8_t4_packed u_arg) {
  // CHECK: %{{.*}} = alloca i32, align 4
  int8_t4_packed s1;
  // CHECK: %{{.*}} = alloca [10 x i32], align 4
  int8_t4_packed s2[10];
  // CHECK: %{{.*}} = alloca i32, align 4
  uint8_t4_packed u1;
  // CHECK: %{{.*}} = alloca [10 x i32], align 4
  uint8_t4_packed u2[10];

  // CHECK: [[LOAD:%.*]] = load i32, ptr
  // CHECK-NEXT: store i32 [[LOAD]], ptr
  uint a = s_arg;
  // CHECK-NEXT: [[LOAD:%.*]] = load i32, ptr
  // CHECK-NEXT: store i32 [[LOAD]], ptr
  uint b = u_arg;
  // CHECK-NEXT: [[LOAD:%.*]] = load i32, ptr
  // CHECK-NEXT: store i32 [[LOAD]], ptr
  uint c = (uint)s_arg;
  // CHECK-NEXT: [[LOAD:%.*]] = load i32, ptr
  // CHECK-NEXT: store i32 [[LOAD]], ptr
  uint d = (uint)u_arg;
  // CHECK-NEXT: [[LOAD:%.*]] = load i32, ptr
  // CHECK-NEXT: store i32 [[LOAD]], ptr
  int8_t4_packed u_to_s = u_arg;
  // CHECK-NEXT: [[LOAD:%.*]] = load i32, ptr
  // CHECK-NEXT: store i32 [[LOAD]], ptr
  uint8_t4_packed s_to_u = s_arg;
}
