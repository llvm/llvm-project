// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -std=c++11 -include-pch %t -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

struct vector {
  vector() = default;
  int at(int) { return 0; }
  int at(int) const { return 1; }
};

// CHECK: template <typename T> void test(const vector begin_vec) {
// CHECK:    #pragma omp parallel for collapse(2)
// CHECK:        for (int n = begin_vec.at(0); n < 0; n++) {
// CHECK:            for (int h = begin_vec.at(1); h < 1; h++) {
// CHECK:                ;
// CHECK:            }
// CHECK:        }
// CHECK: }
// CHECK: template<> void test<int>(const vector begin_vec) {
// CHECK:    #pragma omp parallel for collapse(2)
// CHECK:        for (int n = begin_vec.at(0); n < 0; n++) {
// CHECK:            for (int h = begin_vec.at(1); h < 1; h++) {
// CHECK:                ;
// CHECK:            }
// CHECK:        }
// CHECK: }
template <typename T>
void test(const vector begin_vec) {
#pragma omp parallel for collapse(2)
  for (int n = begin_vec.at(0); n < 0; n++) {
    for (int h = begin_vec.at(1); h < 1; h++) {
      ;
    }
  }
}

int main() {
  vector v;
  test<int>(v);
}
#endif
