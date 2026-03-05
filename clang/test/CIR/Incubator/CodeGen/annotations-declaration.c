// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

__attribute__((annotate("bar"))) int foo();

int main() {
  return foo();
}

// CIR: module {{.*}}annotations-declaration.c" attributes
// CIR-SAME: {cir.global_annotations = #cir<global_annotations [
// CIR-SAME: ["foo", #cir.annotation<name = "bar", args = []>]

// LLVM: target triple
// LLVM-DAG: private unnamed_addr constant [4 x i8] c"bar\00", section "llvm.metadata"

// LLVM: @llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }] [{
// LLVM-SAME: { ptr @foo,
// LLVM-SAME: }], section "llvm.metadata"

// OGCG: target triple
// OGCG-DAG: private unnamed_addr constant [4 x i8] c"bar\00", section "llvm.metadata"

// OGCG: @llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }] [{
// OGCG-SAME: { ptr @foo,
// OGCG-SAME: }], section "llvm.metadata"
