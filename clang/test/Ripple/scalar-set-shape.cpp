// REQUIRES: target=hexagon{{.*}}  || target-aarch64 || target-x86_64
// RUN: %clang %s -O2 -fenable-ripple -S -emit-llvm -o - | FileCheck %s

#include <ripple.h>

enum {
  I = 1,
  J = 1,
};

struct X {
  float A[I];
};

void run(X &x, float B[J], float C[I][J]) {
  auto BS = ripple_set_block_shape(0, J, I);
  auto i = ripple_id(BS, 1);
  auto j = ripple_id(BS, 0);
  C[i][j] = ripple_broadcast(BS, 0xffff, x.A[i]) + ripple_reduceadd(0xfff, B[j]);
  C[0][0] += ripple_get_block_size(BS, 1) + ripple_get_block_size(BS, 0);
}

// CHECK: define dso_local void @{{.*}}run{{.*}}(ptr{{.*}}[[x:%.*]], ptr{{.*}}[[B:%.*]], ptr{{.*}}[[C:%.*]]){{.*}}{
// CHECK: [[XLoad:%.*]] = load float, ptr [[x]]
// CHECK-NEXT: [[BLoad:%.*]] = load float, ptr [[B]]
// CHECK-NEXT: [[Add:%.*]] = fadd float [[XLoad]], [[BLoad]]
// CHECK-NEXT: [[Add2:%.*]] = fadd float [[Add]], 2.0
// CHECK-NEXT: store float [[Add2]], ptr [[C]]
