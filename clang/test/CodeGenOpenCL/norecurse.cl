// RUN: %clang_cc1 -O0 -emit-llvm -o - %s | FileCheck %s

kernel void kernel1(int a) {}
// CHECK: define{{.*}}@kernel1{{.*}}#[[ATTR:[0-9]*]]

// CHECK: attributes #[[ATTR]] = {{.*}}norecurse
