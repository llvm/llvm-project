// RUN: %clang_cc1 -emit-llvm -o - -fcuda-is-device -x hip %s | FileCheck %s

// CHECK-NOT: llvm.linker.options
#pragma comment(lib, "a.so")
