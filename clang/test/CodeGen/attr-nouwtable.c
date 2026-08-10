// RUN: %clang_cc1 -funwind-tables=2 -emit-llvm %s -o - | FileCheck %s

__attribute__((nouwtable))
int test1(void) { return 0; }

// CHECK: @test1{{.*}}[[ATTR1:#[0-9]+]]
// CHECK: attributes [[ATTR1]] = {
// CHECK-NOT: uwtable
// CHECK: }
