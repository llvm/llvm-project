// RUN: %clang_cc1 -fopenmp -std=c++20 -triple x86_64-unknown-linux-gnu \
// RUN:   -ast-dump %s | FileCheck %s


struct Point {
  int x, y;
};

void test_target_firstprivate() {
  Point p{1, 2};
  auto [a, b] = p;
#pragma omp target firstprivate(a)
  {
    a = 1;
  }
}

// CHECK-LABEL: test_target_firstprivate
// CHECK: OMPCaptureKindAttr {{.*}} firstprivate
// CHECK-NOT: OMPCaptureKindAttr {{.*}} map

void test_target_parallel_firstprivate() {
  Point p{1, 2};
  auto [a, b] = p;
#pragma omp target parallel firstprivate(a)
  {
    a = 1;
  }
}

// CHECK-LABEL: test_target_parallel_firstprivate
// CHECK: OMPCaptureKindAttr {{.*}} firstprivate
// CHECK-NOT: OMPCaptureKindAttr {{.*}} map 
