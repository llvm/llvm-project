// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa -fopenmp -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

struct S {
  int x;
  int *arr;
  int y;
  int z;
};

int main() {
  S v;

#pragma omp target map(tofrom: v, v.x, v.z)
  {
    v.x++;
    v.y += 2;
    v.z += 3;
  }

#pragma omp target map(tofrom: v, v.x, v.arr[:1])
  {
    v.x++;
    v.y += 2;
    v.arr[0] += 2;
    v.z += 4;
  }

#pragma omp target map(tofrom: v, v.arr[:1])
  {
    v.x++;
    v.y += 2;
    v.arr[0] += 2;
    v.z += 4;
  }

  return 0;
}

// CHECK: [[CSTSZ0:@.+]] = private {{.*}}constant [4 x i64] [i64 0, i64 0, i64 4, i64 4]
// CHECK: [[CSTTY0:@.+]] = private {{.*}}constant [4 x i64] [i64 [[#0x20]], i64 [[#0x1000000000003]], i64 [[#0x1000000000003]], i64 [[#0x1000000000003]]]

// CHECK: [[CSTSZ1:@.+]] = private {{.*}}constant [4 x i64] [i64 0, i64 0, i64 4, i64 4]
// CHECK: [[CSTTY1:@.+]] = private {{.*}}constant [4 x i64] [i64 [[#0x20]], i64 [[#0x1000000000003]], i64 [[#0x1000000000003]], i64 [[#0x1000000000013]]]

// CHECK: [[CSTSZ2:@.+]] = private {{.*}}constant [3 x i64] [i64 0, i64 24, i64 4]
// CHECK: [[CSTTY2:@.+]] = private {{.*}}constant [3 x i64] [i64 [[#0x20]], i64 [[#0x1000000000003]], i64 [[#0x1000000000013]]]

// CHECK-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CHECK-DAG: [[KSIZE:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 4
// CHECK-DAG: store ptr [[SZBASE:%.+]], ptr [[KSIZE]], align 8
// CHECK-DAG: [[SZBASE]] = getelementptr inbounds [4 x i64], ptr [[SIZES:%[^,]*]], i32 0, i32 0

// Fill two non-constant size elements here: the whole struct size, and the
// region covering v.arr and v.y.

// CHECK-DAG: [[STR:%.+]] = getelementptr inbounds [4 x i64], ptr [[SIZES]], i32 0, i32 0
// CHECK-DAG: store i64 %{{.+}}, ptr [[STR]], align 8
// CHECK-DAG: [[ARRY:%.+]] = getelementptr inbounds [4 x i64], ptr [[SIZES]], i32 0, i32 1
// CHECK-DAG: store i64 %{{.+}}, ptr [[ARRY]], align 8

// CHECK: call void

// CHECK-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CHECK-DAG: [[KSIZE:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 4
// CHECK-DAG: store ptr [[SZBASE:%.+]], ptr [[KSIZE]], align 8
// CHECK-DAG: [[SZBASE]] = getelementptr inbounds [4 x i64], ptr [[SIZES:%[^,]*]], i32 0, i32 0

// Fill two non-constant size elements here: the whole struct size, and the
// region covering v.arr, v.y and v.z.

// CHECK-DAG: [[STR:%.+]] = getelementptr inbounds [4 x i64], ptr [[SIZES]], i32 0, i32 0
// CHECK-DAG: store i64 %{{.+}}, ptr [[STR]], align 8
// CHECK-DAG: [[ARRYZ:%.+]] = getelementptr inbounds [4 x i64], ptr [[SIZES]], i32 0, i32 1
// CHECK-DAG: store i64 %{{.+}}, ptr [[ARRYZ]], align 8

// CHECK: call void

// CHECK-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CHECK-DAG: [[KSIZE:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 4
// CHECK-DAG: store ptr [[SZBASE:%.+]], ptr [[KSIZE]], align 8
// CHECK-DAG: [[SZBASE]] = getelementptr inbounds [3 x i64], ptr [[SIZES:%[^,]*]], i32 0, i32 0

// Fill one non-constant size element here: the whole struct size.

// CHECK-DAG: [[STR:%.+]] = getelementptr inbounds [3 x i64], ptr [[SIZES]], i32 0, i32 0
// CHECK-DAG: store i64 %{{.+}}, ptr [[STR]], align 8
