// RUN: %clang_cc1 -std=c++11 -triple amdgcn-amd-amdhsa \
// RUN:   -fcuda-is-device -emit-llvm -o - -x hip %s | FileCheck %s

#include "Inputs/cuda.h"

struct A {
  int x;
};

constexpr int constexpr_var = 1;
constexpr A constexpr_struct{2};
constexpr A constexpr_array[4] = {0, 0, 0, 3};
constexpr char constexpr_str[] = "abcd";
const int const_var = 4;

// CHECK-DAG: @_ZL13constexpr_str.const = private unnamed_addr addrspace(4) constant [5 x i8] c"abcd\00"
// CHECK-DAG: @_ZL13constexpr_var = internal addrspace(4) constant i32 1
// CHECK-DAG: @_ZL16constexpr_struct = internal addrspace(4) constant %struct.A { i32 2 }
// CHECK-DAG: @_ZL15constexpr_array = internal addrspace(4) constant [4 x %struct.A] [%struct.A zeroinitializer, %struct.A zeroinitializer, %struct.A zeroinitializer, %struct.A { i32 3 }]
// CHECK-NOT: external

// CHECK-LABEL: define{{.*}}@_Z7dev_funPiPPKi
// CHECK: store i32 1
// CHECK: store i32 2
// CHECK: store i32 3
// CHECK: store i32 4
// CHECK: load i8, i8* getelementptr {{.*}} @_ZL13constexpr_str.const
// CHECK: store i32* {{.*}}@_ZL13constexpr_var
// CHECK: store i32* getelementptr {{.*}} @_ZL16constexpr_struct
// CHECK: store i32* getelementptr {{.*}} @_ZL15constexpr_array
__device__ void dev_fun(int *out, const int **out2) {
  *out = constexpr_var;
  *out = constexpr_struct.x;
  *out = constexpr_array[3].x;
  *out = const_var;
  *out = constexpr_str[3];
  *out2 = &constexpr_var;
  *out2 = &constexpr_struct.x;
  *out2 = &constexpr_array[3].x;
}
