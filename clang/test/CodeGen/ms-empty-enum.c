// RUN: %clang_cc1 -fms-extensions -triple x86_64-windows-msvc -Wno-implicit-function-declaration -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fms-extensions -triple i386-windows-msvc -Wno-implicit-function-declaration -emit-llvm %s -o - | FileCheck %s

typedef enum tag1 {} A;

// CHECK: void @empty_enum(i32 noundef %a)
void empty_enum(A a) {}
