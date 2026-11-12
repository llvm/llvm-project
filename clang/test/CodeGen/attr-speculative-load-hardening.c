// RUN: %clang_cc1 -mspeculative-load-hardening -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s -check-prefix=SLH
// RUN: %clang -mno-speculative-load-hardening -S -emit-llvm %s -o - | FileCheck %s -check-prefix=NOSLH
//
// Check that we set the attribute on each function.

int test1(void) {
  return 42;
}
// SLH: @{{.*}}test1{{.*}}[[SLH:#[0-9]+]]

// SLH: attributes [[SLH]] = { {{.*}}speculative_load_hardening{{.*}} }

// NOSLH: @{{.*}}test1{{.*}}[[NOSLH:#[0-9]+]]

// NOSLH-NOT: attributes [[NOSLH]] = { {{.*}}speculative_load_hardening{{.*}} }
