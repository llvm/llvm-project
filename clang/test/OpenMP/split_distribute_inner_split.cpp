// `distribute` outer loop with inner `split` (combined-construct interop beyond host `teams` case).
//
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -x c++ -fopenmp -fopenmp-version=60 -O0 -emit-llvm %s -o - | FileCheck %s

// CHECK-LABEL: define {{.*}} @f(
// CHECK: .split.iv
extern "C" void f(void) {
#pragma omp distribute
  for (int i = 0; i < 10; ++i) {
#pragma omp split counts(2, omp_fill)
    for (int j = 0; j < 10; ++j) {
    }
  }
}
