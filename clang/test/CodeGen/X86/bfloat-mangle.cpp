// RUN: %clang_cc1 -triple i386-unknown-unknown -target-feature +sse2 -emit-llvm -o - %s | FileCheck %s --check-prefixes=LINUX
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-feature +sse2 -emit-llvm -o - %s | FileCheck %s --check-prefixes=LINUX
// RUN: %clang_cc1 -triple i386-windows-msvc -target-feature +sse2 -emit-llvm -o - %s | FileCheck %s --check-prefixes=WINDOWS
// RUN: %clang_cc1 -triple x86_64-windows-msvc -target-feature +sse2 -emit-llvm -o - %s | FileCheck %s --check-prefixes=WINDOWS

// LINUX: define {{.*}}void @_Z3fooDF16b(bfloat noundef %b)
// WINDOWS: define {{.*}}void @"?foo@@YAXU__bf16@__clang@@@Z"(bfloat noundef %b)
void foo(__bf16 b) {}
