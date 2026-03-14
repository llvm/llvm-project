// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 %s -x c++ -fopenmp -fsyntax-only -verify=host
// host-no-diagnostics

// RUN: %clang_cc1 %s -x hip -fopenmp -fsyntax-only -verify=device
// device-error@#01 {{HIP does not support OpenMP target directives}}
// device-error@#02 {{HIP does not support OpenMP target directives}}
// device-error@#03 {{HIP does not support OpenMP target directives}}
// device-error@#04 {{HIP does not support OpenMP target directives}}
// device-error@#05 {{HIP does not support OpenMP target directives}}
// device-error@#06 {{HIP does not support OpenMP target directives}}
// device-error@#07 {{HIP does not support OpenMP target directives}}
// device-error@#08 {{HIP does not support OpenMP target directives}}
// device-error@#09 {{HIP does not support OpenMP target directives}}
// device-error@#10 {{HIP does not support OpenMP target directives}}
// device-error@#11 {{HIP does not support OpenMP target directives}}
// device-error@#12 {{HIP does not support OpenMP target directives}}
// device-error@#13 {{HIP does not support OpenMP target directives}}
// device-error@#14 {{HIP does not support OpenMP target directives}}
// device-error@#15 {{HIP does not support OpenMP target directives}}
// device-error@#16 {{HIP does not support OpenMP target directives}}
// device-error@#17 {{HIP does not support OpenMP target directives}}
// device-error@#18 {{HIP does not support OpenMP target directives}}
// device-error@#19 {{HIP does not support OpenMP target directives}}
// device-error@#20 {{HIP does not support OpenMP target directives}}
// device-error@#21 {{HIP does not support OpenMP target directives}}
// device-error@#22 {{HIP does not support OpenMP target directives}}
// device-error@#23 {{HIP does not support OpenMP target directives}}
// device-error@#24 {{HIP does not support OpenMP target directives}}

void test01() {
#pragma omp target // #01
  ;
}


void test02() {
#pragma omp target parallel // #02
  ;
}

void test03() {
#pragma omp target parallel for // #03
  for (int i = 0; i < 1; ++i);
}

void test04(int x) {
#pragma omp target data map(x) // #04
  ;
}

void test05(int * x, int n) {
#pragma omp target enter data map(to:x[:n]) // #05
}

void test06(int * x, int n) {
#pragma omp target exit data map(from:x[:n]) // #06
}

void test07(int * x, int n) {
#pragma omp target update to(x[:n]) // #07
}

#pragma omp declare target (test07) // #08
void test08() {

}

#pragma omp begin declare target // #09
void test09_1() {

}

void test09_2() {

}
#pragma omp end declare target

void test10(int n) {
  #pragma omp target parallel // #10
  for (int i = 0; i < n; ++i)
    ;
}

void test11(int n) {
  #pragma omp target parallel for // #11
  for (int i = 0; i < n; ++i)
    ;
}

void test12(int n) {
  #pragma omp target parallel for simd // #12
  for (int i = 0; i < n; ++i)
    ;
}

void test13(int n) {
  #pragma omp target parallel loop // #13
  for (int i = 0; i < n; ++i)
    ;
}

void test14(int n) {
  #pragma omp target simd // #14
  for (int i = 0; i < n; ++i)
    ;
}

void test15(int n) {
  #pragma omp target teams // #15
  for (int i = 0; i < n; ++i)
    ;
}

void test16(int n) {
  #pragma omp target teams distribute // #16
  for (int i = 0; i < n; ++i)
    ;
}

void test17(int n) {
  #pragma omp target teams distribute simd // #17
  for (int i = 0; i < n; ++i)
    ;
}

void test18(int n) {
  #pragma omp target teams loop // #18
  for (int i = 0; i < n; ++i)
    ;
}

void test19(int n) {
  #pragma omp target teams distribute parallel for // #19
  for (int i = 0; i < n; ++i)
    ;
}

void test20(int n) {
  #pragma omp target teams distribute parallel for simd // #20
  for (int i = 0; i < n; ++i)
    ;
}

void test21() {
#pragma omp target // #21
  {
#pragma omp teams // #22
    {}
  }
}

void test22() {
#pragma omp target // #23
#pragma omp teams // #24
  {}
}

void test23() {
// host code
#pragma omp teams
  {}
}
