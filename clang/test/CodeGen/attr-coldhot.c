// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s -check-prefixes=CHECK,O0
// RUN: %clang_cc1 -emit-llvm %s -o - -O1 -disable-llvm-passes | FileCheck %s -check-prefixes=CHECK,O1

int test1(void) __attribute__((__cold__)) {
  return 42;

// Check that we set the optsize attribute on the function.
// CHECK: @test1{{.*}}[[ATTR:#[0-9]+]]
// CHECK: ret
}

// O0: attributes [[ATTR]] = { {{.*}}cold{{.*}}optnone{{.*}} }
// O1: attributes [[ATTR]] = { {{.*}}cold{{.*}}optsize{{.*}} }
