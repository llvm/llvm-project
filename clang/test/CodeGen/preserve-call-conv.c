// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm < %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-unknown-unknown -emit-llvm < %s | FileCheck %s

// Check that the preserve_most calling convention attribute at the source level
// is lowered to the corresponding calling convention attrribute at the LLVM IR
// level.
void foo() __attribute__((preserve_most)) {
  // CHECK-LABEL: define preserve_mostcc void @foo()
}

// Check that the preserve_most calling convention attribute at the source level
// is lowered to the corresponding calling convention attrribute at the LLVM IR
// level.
void boo() __attribute__((preserve_all)) {
  // CHECK-LABEL: define preserve_allcc void @boo()
}

