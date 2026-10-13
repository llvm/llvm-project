// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -x c++ -emit-llvm %s -triple x86_64-unknown-unknown -o - | FileCheck %s
// expected-no-diagnostics

// CHECK-LABEL: @_Z4testii(
// CHECK: [[CMP:%.*]] = icmp sgt i32 %{{.*}}, 0
// CHECK: call void @llvm.assume(i1 [[CMP]])
void test(int n, int m) {
  #pragma omp assume holds(n > 0)
  {
    m = n + 1;
  }
}

// CHECK-LABEL: @_Z12test_complexi(
// CHECK: [[CMP1:%.*]] = icmp sge i32 %{{.*}}, 10
// CHECK: br i1 [[CMP1]]
// CHECK: [[CMP2:%.*]] = icmp eq i32 %{{.*}}, 0
// CHECK: [[COND:%.*]] = phi i1 [ false, %{{.*}} ], [ [[CMP2]], %{{.*}} ]
// CHECK: call void @llvm.assume(i1 [[COND]])
void test_complex(int n) {
  #pragma omp assume holds(n >= 10 && n % 4 == 0)
  {
    for (int i = 0; i < n; i++) {}
  }
}

// CHECK-LABEL: @_Z16test_multi_holdsii(
// CHECK: [[CMPA:%.*]] = icmp ne i32 %{{.*}}, 0
// CHECK: call void @llvm.assume(i1 [[CMPA]])
// CHECK: [[CMPB:%.*]] = icmp slt i32 %{{.*}}, 100
// CHECK: call void @llvm.assume(i1 [[CMPB]])
void test_multi_holds(int a, int b) {
  #pragma omp assume holds(a) holds(b < 100)
  {
    int c = a + b;
  }
}

// CHECK-LABEL: @_Z7test_orii(
// CHECK: [[CMPA:%.*]] = icmp eq i32 %{{.*}}, 1
// CHECK: br i1 [[CMPA]]
// CHECK: [[CMPB:%.*]] = icmp eq i32 %{{.*}}, 2
// CHECK: [[COND:%.*]] = phi i1 [ true, %{{.*}} ], [ [[CMPB]], %{{.*}} ]
// CHECK: call void @llvm.assume(i1 [[COND]])
void test_or(int mode, int x) {
  #pragma omp assume holds(mode == 1 || mode == 2)
  {
    x = mode + 1;
  }
}
