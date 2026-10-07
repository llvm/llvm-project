// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -x c++ -emit-llvm %s -triple x86_64-unknown-unknown -o - | FileCheck %s

int g;
int f() { return g++; }

void test_side_effects() {
  // expected-warning@+1 {{assumption is ignored because it contains (potential) side-effects}}
  #pragma omp assume holds(f())
  {
    g = 1;
  }
}

// CHECK-LABEL: @_Z17test_side_effectsv(
// CHECK-NOT: call void @llvm.assume
// CHECK-NOT: call{{.*}}@_Z1fv
// CHECK: store i32 1
// CHECK: ret void

void test_mixed(int n, int m) {
  // expected-warning@+1 {{assumption is ignored because it contains (potential) side-effects}}
  #pragma omp assume holds(n > 0) holds(m++)
  {
    g = n;
  }
}

// CHECK-LABEL: @_Z10test_mixedii(
// CHECK: [[CMP:%.*]] = icmp sgt i32 %{{.*}}, 0
// CHECK: call void @llvm.assume(i1 [[CMP]])
// CHECK-NOT: call void @llvm.assume(i1
// CHECK: ret void
