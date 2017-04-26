// RUN: %clang_cc1 -fsanitize=address -emit-llvm -o - -triple x86_64-windows-msvc %s | FileCheck %s --check-prefix=WITH-GC
// RUN: %clang_cc1 -fsanitize=address -emit-llvm -o - -triple x86_64-windows-msvc -fdata-sections %s | FileCheck %s --check-prefix=WITH-GC

int global;

// WITH-GC-NOT: call void @__asan_register_globals
// WITHOUT-GC: call void @__asan_register_globals
