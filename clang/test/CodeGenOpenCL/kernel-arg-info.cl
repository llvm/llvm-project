// RUN: %clang_cc1 %s -cl-kernel-arg-info -emit-llvm -o - | FileCheck %s

kernel void foo(int *X, int Y, int anotherArg) {
  *X = Y + anotherArg;
}

// CHECK: metadata !{metadata !"kernel_arg_name", metadata !"X", metadata !"Y", metadata !"anotherArg"}
