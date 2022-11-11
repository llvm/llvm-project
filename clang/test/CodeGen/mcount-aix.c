// RUN: %clang_cc1 -pg -triple powerpc-ibm-aix7.2.0.0 -S -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -pg -triple powerpc64-ibm-aix7.2.0.0 -S -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK64

void foo() {
}

void bar() {
    foo();
}
// CHECK: @[[GLOB0:[0-9]+]] = internal global i32 0
// CHECK: @[[GLOB1:[0-9]+]] = internal global i32 0
// CHECK64: @[[GLOB0:[0-9]+]] = internal global i64 0
// CHECK64: @[[GLOB1:[0-9]+]] = internal global i64 0
// CHECK-LABEL: @foo(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    call void @__mcount(ptr @[[GLOB0]])
// CHECK64-LABEL: @foo(
// CHECK64-NEXT:  entry:
// CHECK64-NEXT:    call void @__mcount(ptr @[[GLOB0]])
// CHECK-LABEL: @bar(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    call void @__mcount(ptr @[[GLOB1]])
// CHECK64-LABEL: @bar(
// CHECK64-NEXT:  entry:
// CHECK64-NEXT:    call void @__mcount(ptr @[[GLOB1]])
