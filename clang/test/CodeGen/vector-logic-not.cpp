// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s

typedef __attribute__((__vector_size__(16))) float float4;
typedef __attribute__((__vector_size__(16))) int int4;
typedef __attribute__((__vector_size__(16))) unsigned int uint4;

// CHECK: @_Z5test1Dv4_j
int4 test1(uint4 V0) {
  // CHECK: [[CMP0:%.*]] = icmp eq <4 x i32> [[V0:%.*]], zeroinitializer
  // CHECK-NEXT: [[V1:%.*]] = sext <4 x i1> [[CMP0]] to <4 x i32>
  int4 V = !V0;
  return V;
}

// CHECK: @_Z5test2Dv4_fS_
int4 test2(float4 V0, float4 V1) {
  // CHECK: [[CMP0:%.*]] = fcmp oeq <4 x float> [[V0:%.*]], zeroinitializer
  // CHECK-NEXT: [[V1:%.*]] = sext <4 x i1> [[CMP0]] to <4 x i32>
  int4 V = !V0;
  return V;
}
