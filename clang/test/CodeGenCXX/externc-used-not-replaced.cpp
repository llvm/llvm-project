// RUN: %clang_cc1 -triple x86_64-windows -emit-llvm -o - %s | FileCheck %s

extern "C" {
  const char a __attribute__((used)){};
}

// CHECK: @a = internal constant i8 0
// CHECK: @llvm.used = appending global [1 x ptr] [ptr @a]
