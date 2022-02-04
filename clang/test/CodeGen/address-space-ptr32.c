// RUN: %clang_cc1 -triple x86_64-windows-msvc -fms-extensions -emit-llvm < %s | FileCheck %s

int foo(void) {
  int (*__ptr32 a)(int);
  return sizeof(a);
}

// CHECK: define dso_local i32 @foo
// CHECK: %a = alloca i32 (i32) addrspace(270)*, align 4
// CHECK: ret i32 4
