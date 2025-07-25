// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa -fopenmp -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

struct S {
  int x;
  int y;
  int z;
  int *p1;
  int *p2;
};

struct T : public S {
  int a;
  int b;
  int c;
};

int main() {
  T v;

#pragma omp target map(tofrom: v, v.x, v.y, v.z, v.p1[:8], v.a, v.b, v.c)
  {
    v.x++;
    v.y += 2;
    v.z += 3;
    v.p1[0] += 4;
    v.a += 7;
    v.b += 5;
    v.c += 6;
  }

  return 0;
}

// CHECK: [[CSTSZ:@.+]] = private {{.*}}constant [10 x i64] [i64 0, i64 0, i64 0, i64 4, i64 4, i64 4, i64 32, i64 4, i64 4, i64 4]
// CHECK: [[CSTTY:@.+]] = private {{.*}}constant [10 x i64] [i64 [[#0x20]], i64 [[#0x1000000000003]], i64 [[#0x1000000000003]], i64 [[#0x1000000000003]], i64 [[#0x1000000000003]], i64 [[#0x1000000000003]], i64 [[#0x1000000000013]], i64 [[#0x1000000000003]], i64 [[#0x1000000000003]], i64 [[#0x1000000000003]]]

// CHECK-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CHECK-DAG: [[KSIZE:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 4
// CHECK-DAG: store ptr [[SZBASE:%.+]], ptr [[KSIZE]], align 8
// CHECK-DAG: [[SZBASE]] = getelementptr inbounds [10 x i64], ptr [[SIZES:%[^,]*]], i32 0, i32 0

// Check for filling of four non-constant size elements here: the whole struct
// size, the (padded) region covering p1 & p2, and the padding at the end of
// struct T.

// CHECK-DAG: [[STR:%.+]] = getelementptr inbounds [10 x i64], ptr [[SIZES]], i32 0, i32 0
// CHECK-DAG: store i64 %{{.+}}, ptr [[STR]], align 8
// CHECK-DAG: [[P1P2:%.+]] = getelementptr inbounds [10 x i64], ptr [[SIZES]], i32 0, i32 1
// CHECK-DAG: store i64 %{{.+}}, ptr [[P1P2]], align 8
// CHECK-DAG: [[PAD:%.+]] = getelementptr inbounds [10 x i64], ptr [[SIZES]], i32 0, i32 2
// CHECK-DAG: store i64 %{{.+}}, ptr [[PAD]], align 8
