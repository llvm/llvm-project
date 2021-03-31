// RUN: %clang_cc1 -verify -fopenmp -triple x86_64-apple-darwin10.6.0 -emit-llvm -o - %s 2>&1 | FileCheck %s
// expected-no-diagnostics

// CHECK-NOT: fatal error: cannot open file

// CHECK: call void @__omp_offloading_{{.+}}()
# 1 "unknown.xxxxxxxx"
void a() {
#pragma omp target
  ;
}

// CHECK-NOT: fatal error: cannot open file
