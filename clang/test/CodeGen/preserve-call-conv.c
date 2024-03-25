// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm < %s | FileCheck %s --check-prefixes=CHECK,X86-LINUX
// RUN: %clang_cc1 -triple arm64-unknown-unknown -emit-llvm < %s | FileCheck %s

// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-unknown-windows-msvc -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple thumbv7-unknown-windows-msvc -emit-llvm %s -o - | FileCheck %s

// Check that the preserve_most calling convention attribute at the source level
// is lowered to the corresponding calling convention attrribute at the LLVM IR
// level.
void foo(void) __attribute__((preserve_most)) {
  // CHECK-LABEL: define {{(dso_local )?}}preserve_mostcc void @foo()
}

// Check that the preserve_most calling convention attribute at the source level
// is lowered to the corresponding calling convention attrribute at the LLVM IR
// level.
void boo(void) __attribute__((preserve_all)) {
  // CHECK-LABEL: define {{(dso_local )?}}preserve_allcc void @boo()
}

// Check that the preserve_none calling convention attribute at the source level
// is lowered to the corresponding calling convention attrribute at the LLVM IR
// level.
void bar(void) __attribute__((preserve_none)) {
  // X86-LINUX-LABEL: define {{(dso_local )?}}preserve_nonecc void @bar()
}
