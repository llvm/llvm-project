// RUN: %clang_cc1  -fvisibility=default -emit-llvm -o - %s | FileCheck %s

// CHECK-DAG: @g0 = external hidden constant i32
// CHECK-DAG: @g1 = hidden constant i32 1

__private_extern__ const int g0;
__private_extern__ const int g1 = 1;

int f0(void) {
  return g0;
}
