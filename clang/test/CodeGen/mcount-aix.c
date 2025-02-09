// RUN: %clang_cc1 -disable-llvm-passes -pg -triple powerpc-ibm-aix7.2.0.0 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -disable-llvm-passes -pg -triple powerpc64-ibm-aix7.2.0.0 -emit-llvm %s -o - | FileCheck %s

void foo() {
// CHECK: define void @foo() #0 {
}

void bar() {
// CHECK: define void @bar() #0 {
    foo();
}

// CHECK: attributes #0 = { {{.*}}"instrument-function-entry-inlined"="__mcount"
