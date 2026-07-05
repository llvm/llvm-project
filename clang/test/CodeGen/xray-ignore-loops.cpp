// RUN: %clang_cc1 -fxray-instrument -fxray-ignore-loops -x c++ -std=c++11 -emit-llvm -o - %s -triple x86_64-unknown-linux-gnu | FileCheck %s

int foo() {
  return 1;
}

// CHECK: define{{.*}} i32 @_Z3foov() #[[ATTRS:[0-9]+]] {
// CHECK-DAG: attributes #[[ATTRS]] = {{.*}} "xray-ignore-loops" {{.*}}
