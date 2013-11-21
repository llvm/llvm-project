// RUN: %clang_cc1 -triple nvptx-unknown-unknown -S -o - %s -emit-llvm | FileCheck %s
// RUN: %clang_cc1 -triple nvptx64-unknown-unknown -S -o - %s -emit-llvm | FileCheck %s

typedef struct float4_s {
  float x, y, z, w;
} float4_t;

float4_t my_function(void);

// CHECK-DAG: declare %struct.float4_s @my_function

float bar(void) {
  float4_t ret;
// CHECK-DAG: call %struct.float4_s @my_function
  ret = my_function();
  return ret.x;
}
