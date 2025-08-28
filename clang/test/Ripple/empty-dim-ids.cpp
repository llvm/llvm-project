// REQUIRES: target=hexagon{{.*}} || target-aarch64 || target-x86_64
// RUN: %clang %s -O2 -fenable-ripple -S -emit-llvm -o - | FileCheck %s

#include <ripple.h>

enum {
  I = 1,
  J = 8,
};

struct X {
  float A[I];
};

void check(X &x, float B[J], float C[I][J]) {
  auto BS = ripple_set_block_shape(0, J, I);
  auto i = ripple_id(BS, 1);
  auto j = ripple_id(BS, 0);
  C[i][j] = x.A[i] + B[j];
  C[0][0] += ripple_get_block_size(BS, 1) + ripple_get_block_size(BS, 0);
}

// CHECK: define dso_local void @{{.*}}check{{.*}}(ptr{{.*}}[[x:%.*]], ptr{{.*}}[[B:%.*]], ptr{{.*}}[[C:%.*]]){{.*}}{
// CHECK: [[XLoad:%.*]] = load float, ptr [[x]]
// CHECK-NEXT: [[XInsert:%.*]] = insertelement <8 x float> poison, float [[XLoad]]
// CHECK-NEXT: [[XSplat:%.*]] = shufflevector <8 x float> [[XInsert]], <8 x float> poison, <8 x i32> zeroinitializer
// CHECK-NEXT: [[BLoad:%.*]] = load <8 x float>, ptr [[B]]
// CHECK-NEXT: [[Add:%.*]] = fadd <8 x float> [[XSplat]], [[BLoad]]
// CHECK-NEXT: store <8 x float> [[Add]], ptr [[C]]
