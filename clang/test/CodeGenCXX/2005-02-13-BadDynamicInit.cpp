// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s
// This testcase corresponds to PR509
struct Data {
  unsigned *data;
  unsigned array[1];
};

// CHECK-NOT: llvm.global_ctors
Data shared_null = { shared_null.array };
