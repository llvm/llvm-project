// RUN: %clang_cc1 -fxray-instrument -fxray-instruction-threshold=1 -x c++ -std=c++11 -emit-llvm -o - %s -triple x86_64-unknown-linux-gnu | FileCheck %s

int foo() {
  return 1;
}

[[clang::xray_never_instrument]] int bar() {
  return 2;
}

// CHECK: define i32 @_Z3foov() #[[THRESHOLD:[0-9]+]] {
// CHECK: define i32 @_Z3barv() #[[NEVERATTR:[0-9]+]] {
// CHECK-DAG: attributes #[[THRESHOLD]] = {{.*}} "xray-instruction-threshold"="1" {{.*}}
// CHECK-DAG: attributes #[[NEVERATTR]] = {{.*}} "function-instrument"="xray-never" {{.*}}
