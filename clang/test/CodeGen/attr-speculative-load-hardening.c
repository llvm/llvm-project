// RUN: %clang_cc1 -mspeculative-load-hardening -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s -check-prefix=SLH
//
// Check that we set the attribute on each function.

int test1() {
  return 42;
}
// SLH: @{{.*}}test1{{.*}}[[SLH:#[0-9]+]]

// SLH: attributes [[SLH]] = { {{.*}}speculative_load_hardening{{.*}} }
