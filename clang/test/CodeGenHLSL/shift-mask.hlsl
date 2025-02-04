// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

int shl32(int V, int S) {
  return V << S;
}

// CHECK-LABEL: define noundef i32 @_Z5shl32ii(i32 noundef %V, i32 noundef %S) #0 {
// CHECK-DAG:  %[[Masked:.*]] = and i32 %{{.*}}, 31
// CHECK-DAG:  %{{.*}} = shl i32 %{{.*}}, %[[Masked]]

int shr32(int V, int S) {
  return V >> S;
}

// CHECK-LABEL: define noundef i32 @_Z5shr32ii(i32 noundef %V, i32 noundef %S) #0 {
// CHECK-DAG:  %[[Masked:.*]] = and i32 %{{.*}}, 31
// CHECK-DAG:  %{{.*}} = ashr i32 %{{.*}}, %[[Masked]]

int64_t shl64(int64_t V, int64_t S) {
  return V << S;
}

// CHECK-LABEL: define noundef i64 @_Z5shl64ll(i64 noundef %V, i64 noundef %S) #0 {
// CHECK-DAG:  %[[Masked:.*]] = and i64 %{{.*}}, 63
// CHECK-DAG:  %{{.*}} = shl i64 %{{.*}}, %[[Masked]]

int64_t shr64(int64_t V, int64_t S) {
  return V >> S;
}

// CHECK-LABEL: define noundef i64 @_Z5shr64ll(i64 noundef %V, i64 noundef %S) #0 {
// CHECK-DAG:  %[[Masked:.*]] = and i64 %{{.*}}, 63
// CHECK-DAG:  %{{.*}} = ashr i64 %{{.*}}, %[[Masked]]

uint shlu32(uint V, uint S) {
  return V << S;
}

// CHECK-LABEL: define noundef i32 @_Z6shlu32jj(i32 noundef %V, i32 noundef %S) #0 {
// CHECK-DAG:  %[[Masked:.*]] = and i32 %{{.*}}, 31
// CHECK-DAG:  %{{.*}} = shl i32 %{{.*}}, %[[Masked]]

uint shru32(uint V, uint S) {
  return V >> S;
}

// CHECK-LABEL: define noundef i32 @_Z6shru32jj(i32 noundef %V, i32 noundef %S) #0 {
// CHECK-DAG:  %[[Masked:.*]] = and i32 %{{.*}}, 31
// CHECK-DAG:  %{{.*}} = lshr i32 %{{.*}}, %[[Masked]]

uint64_t shlu64(uint64_t V, uint64_t S) {
  return V << S;
}

// CHECK-LABEL: define noundef i64 @_Z6shlu64mm(i64 noundef %V, i64 noundef %S) #0 {
// CHECK-DAG:  %[[Masked:.*]] = and i64 %{{.*}}, 63
// CHECK-DAG:  %{{.*}} = shl i64 %{{.*}}, %[[Masked]]

uint64_t shru64(uint64_t V, uint64_t S) {
  return V >> S;
}

// CHECK-LABEL: define noundef i64 @_Z6shru64mm(i64 noundef %V, i64 noundef %S) #0 {
// CHECK-DAG:  %[[Masked:.*]] = and i64 %{{.*}}, 63
// CHECK-DAG:  %{{.*}} = lshr i64 %{{.*}}, %[[Masked]]
