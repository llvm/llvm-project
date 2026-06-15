// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -mrelocation-model static \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=STATIC

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -mrelocation-model pic \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=PIC

// STATIC: @.gomp_critical_user_foo.var = common dso_local global [8 x i32] zeroinitializer
// PIC: @.gomp_critical_user_foo.var = common global [8 x i32] zeroinitializer

void f() {
#pragma omp critical(foo)
  ;
}
