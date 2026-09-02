// RUN: %clang_cc1 -verify -fopenmp -O1 -disable-llvm-passes -x c -triple x86_64-unknown-linux-gnu -fopenmp-targets=x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

// The loop bounds used to compute a target region's trip count live in
// '.capture_expr.' allocas created by OMPLoopScope. The scope has to outlive
// the trip count computation, otherwise its cleanups emit lifetime.end before
// the loads and the trip count reads dead memory.

void f(int n, int *a) {
#pragma omp target teams distribute parallel for
  for (int i = 0; i < n; ++i)
    a[i] = i;
}

// CHECK-LABEL: define {{.*}}@f(
// CHECK:      store i32 %{{.+}}, ptr %[[CE:\.capture_expr\.[0-9]+]], align 4
// CHECK-NEXT: [[TC:%.+]] = load i32, ptr %[[CE]], align 4
// CHECK-NEXT: [[ADD:%.+]] = add nsw i32 [[TC]], 1
// CHECK-NEXT: [[EXT:%.+]] = zext i32 [[ADD]] to i64
// CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr %[[CE]])
