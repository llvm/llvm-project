// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa -fopenmp -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

struct S {
  int x;
  int y;
  int z;
};

struct M : public S {
  int mid;
};

struct T : public M {
  int a;
  int b;
  int c;
};

int main() {
  T v;

#pragma omp target map(tofrom: v, v.y, v.z, v.a)
  {
    v.y++;
    v.z += 2;
    v.a += 3;
    v.mid += 5;
  }

  return 0;
}

// CHECK: [[CSTSZ:@.+]] = private {{.*}}constant [7 x i64] [i64 0, i64 0, i64 0, i64 0, i64 4, i64 4, i64 4]
// CHECK: [[CSTTY:@.+]] = private {{.*}}constant [7 x i64] [i64 [[#0x20]], i64 [[#0x1000000000003]], i64 [[#0x1000000000003]], i64 [[#0x1000000000003]], i64 [[#0x1000000000003]], i64 [[#0x1000000000003]], i64 [[#0x1000000000003]]]

// CHECK-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CHECK-DAG: [[KSIZE:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 4
// CHECK-DAG: store ptr [[SZBASE:%.+]], ptr [[KSIZE]], align 8
// CHECK-DAG: [[SZBASE]] = getelementptr inbounds [7 x i64], ptr [[SIZES:%[^,]*]], i32 0, i32 0

// Fill four non-constant size elements here: the whole struct size, the region
// covering v.x, the region covering v.mid and the region covering v.b and v.c.

// CHECK-DAG: [[STR:%.+]] = getelementptr inbounds [7 x i64], ptr [[SIZES]], i32 0, i32 0
// CHECK-DAG: store i64 %{{.+}}, ptr [[STR]], align 8
// CHECK-DAG: [[X:%.+]] = getelementptr inbounds [7 x i64], ptr [[SIZES]], i32 0, i32 1
// CHECK-DAG: store i64 %{{.+}}, ptr [[X]], align 8
// CHECK-DAG: [[MID:%.+]] = getelementptr inbounds [7 x i64], ptr [[SIZES]], i32 0, i32 2
// CHECK-DAG: store i64 %{{.+}}, ptr [[MID]], align 8
// CHECK-DAG: [[BC:%.+]] = getelementptr inbounds [7 x i64], ptr [[SIZES]], i32 0, i32 3
// CHECK-DAG: store i64 %{{.+}}, ptr [[BC]], align 8
