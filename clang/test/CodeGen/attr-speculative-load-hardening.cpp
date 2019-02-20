// RUN: %clang_cc1 -std=c++11 -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK1
// RUN: %clang_cc1 -std=c++11 -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK2
//
// Check that we set the attribute on each function.

[[clang::speculative_load_hardening]]
int test1() {
  return 42;
}

int __attribute__((speculative_load_hardening)) test2() {
  return 42;
}
// CHECK1: @{{.*}}test1{{.*}}[[SLH1:#[0-9]+]]
// CHECK1: attributes [[SLH1]] = { {{.*}}speculative_load_hardening{{.*}} }

// CHECK2: @{{.*}}test2{{.*}}[[SLH2:#[0-9]+]]
// CHECK2: attributes [[SLH2]] = { {{.*}}speculative_load_hardening{{.*}} }
