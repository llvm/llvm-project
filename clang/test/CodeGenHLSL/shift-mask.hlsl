// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

int shl32(int V, int S) {
  return V << S;
}

// CHECK: define noundef i32 @"?shl32{{[@$?.A-Za-z0-9_]+}}"(i32 noundef %V, i32 noundef %S) #0 {
// CHECK-DAG:  %[[Masked:.*]] = and i32 %{{.*}}, 31
// CHECK-DAG:  %{{.*}} = shl i32 %{{.*}}, %[[Masked]]

int shr32(int V, int S) {
  return V >> S;
}

// CHECK: define noundef i32 @"?shr32{{[@$?.A-Za-z0-9_]+}}"(i32 noundef %V, i32 noundef %S) #0 {
// CHECK-DAG:  %[[Masked:.*]] = and i32 %{{.*}}, 31
// CHECK-DAG:  %{{.*}} = ashr i32 %{{.*}}, %[[Masked]]

int64_t shl64(int64_t V, int64_t S) {
  return V << S;
}

// CHECK: define noundef i64 @"?shl64{{[@$?.A-Za-z0-9_]+}}"(i64 noundef %V, i64 noundef %S) #0 {
// CHECK-DAG:  %[[Masked:.*]] = and i64 %{{.*}}, 63
// CHECK-DAG:  %{{.*}} = shl i64 %{{.*}}, %[[Masked]]

int64_t shr64(int64_t V, int64_t S) {
  return V >> S;
}

// CHECK: define noundef i64 @"?shr64{{[@$?.A-Za-z0-9_]+}}"(i64 noundef %V, i64 noundef %S) #0 {
// CHECK-DAG:  %[[Masked:.*]] = and i64 %{{.*}}, 63
// CHECK-DAG:  %{{.*}} = ashr i64 %{{.*}}, %[[Masked]]
