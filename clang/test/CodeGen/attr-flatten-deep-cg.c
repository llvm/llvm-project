// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

// CHECK-LABEL: define {{.*}} @test1
// CHECK-SAME: #[[ATTR1:[0-9]+]]
__attribute__((flatten_deep(3)))
void test1() {
}

// Verify the attribute is present in the attribute groups
// CHECK-DAG: attributes #[[ATTR1]] = { {{.*}}flatten_deep=3{{.*}} }
