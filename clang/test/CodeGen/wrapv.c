// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s --check-prefix=NOWRAP
// RUN: %clang_cc1 -fno-wrapv -emit-llvm %s -o - | FileCheck %s --check-prefix=NOWRAP
// RUN: %clang_cc1 -fno-wrapv -fms-compatibility -emit-llvm %s -o - | FileCheck %s --check-prefix=NOWRAP
// RUN: %clang_cc1 -fwrapv -emit-llvm %s -o - | FileCheck %s --check-prefix=WRAP
// RUN: %clang_cc1 -fms-compatibility -emit-llvm %s -o - | FileCheck %s --check-prefix=WRAP

// NOWRAP: %add = add nsw i32 %0, 1
// WRAP: %add = add i32 %0, 1

int add1(int x) {
  return x + 1;
}
