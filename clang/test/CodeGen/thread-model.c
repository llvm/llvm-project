// Check that the "thread-model" module flag is emitted for -mthread-model
// single, and omitted when the model matches the target default (posix).

// RUN: %clang_cc1 -triple arm-none-linux-gnueabi -mthread-model single -emit-llvm %s -o - | FileCheck %s --check-prefix=SINGLE
// RUN: %clang_cc1 -triple arm-none-linux-gnueabi -mthread-model posix -emit-llvm %s -o - | FileCheck %s --check-prefix=POSIX
// RUN: %clang_cc1 -triple arm-none-linux-gnueabi -emit-llvm %s -o - | FileCheck %s --check-prefix=POSIX

void f(void) {}

// SINGLE: !{i32 1, !"thread-model", !"single"}
// POSIX-NOT: "thread-model"
