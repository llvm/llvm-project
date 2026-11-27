// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-NONE
// RUN: %clang_cc1 -fdenormal-fp-math=ieee %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-NONE
// RUN: %clang_cc1 -fdenormal-fp-math=preserve-sign %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-PS
// RUN: %clang_cc1 -fdenormal-fp-math=positive-zero %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-PZ
// RUN: %clang_cc1 -fdenormal-fp-math=dynamic %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-DYNAMIC

// RUN: %clang_cc1 -fdenormal-fp-math-f32=ieee %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-NONE
// RUN: %clang_cc1 -fdenormal-fp-math=ieee -fdenormal-fp-math-f32=ieee %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-NONE
// RUN: %clang_cc1 -fdenormal-fp-math=preserve-sign -fdenormal-fp-math-f32=ieee %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-PS-F32-IEEE
// RUN: %clang_cc1 -fdenormal-fp-math=positive-zero -fdenormal-fp-math-f32=ieee %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-PZ-F32-IEEE
// RUN: %clang_cc1 -fdenormal-fp-math=positive-zero -fdenormal-fp-math-f32=dynamic %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-PZ-F32-DYNAMIC


// RUN: %clang_cc1 -fdenormal-fp-math-f32=preserve-sign %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-NONE,CHECK-F32-PS
// RUN: %clang_cc1 -fdenormal-fp-math=ieee -fdenormal-fp-math-f32=preserve-sign %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-NONE,CHECK-F32-PS
// RUN: %clang_cc1 -fdenormal-fp-math=preserve-sign -fdenormal-fp-math-f32=preserve-sign %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-PS
// RUN: %clang_cc1 -fdenormal-fp-math=positive-zero -fdenormal-fp-math-f32=preserve-sign %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-PZ-F32-PS
// RUN: %clang_cc1 -fdenormal-fp-math=ieee -fdenormal-fp-math-f32=dynamic %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-NONE,CHECK-F32-DYNAMIC


// RUN: %clang_cc1 -fdenormal-fp-math-f32=positive-zero %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-NONE,CHECK-F32-PZ
// RUN: %clang_cc1 -fdenormal-fp-math-f32=dynamic %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-NONE,CHECK-F32-DYNAMIC
// RUN: %clang_cc1 -fdenormal-fp-math=ieee -fdenormal-fp-math-f32=positive-zero %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-F32-PZ
// RUN: %clang_cc1 -fdenormal-fp-math=dynamic -fdenormal-fp-math-f32=positive-zero %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-DYNAMIC-F32-PZ
// RUN: %clang_cc1 -fdenormal-fp-math=preserve-sign -fdenormal-fp-math-f32=positive-zero %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-PS-F32-PZ
// RUN: %clang_cc1 -fdenormal-fp-math=positive-zero -fdenormal-fp-math-f32=positive-zero %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-PZ
// RUN: %clang_cc1 -fdenormal-fp-math=dynamic -fdenormal-fp-math-f32=dynamic %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-DYNAMIC


// CHECK-LABEL: main

// CHECK-ATTR: attributes #0 =
// CHECK-NONE-NOT: denormal_fpenv

// CHECK-IEEE: denormal_fpenv(ieee)
// CHECK-PS: denormal_fpenv(preservesign)
// CHECK-PZ: denormal_fpenv(positivezero)
// CHECK-DYNAMIC: denormal_fpenv(dynamic)


// CHECK-PS-F32-IEEE: denormal_fpenv(preservesign, float: ieee)
// CHECK-PZ-F32-IEEE: denormal_fpenv(positivezero, float: ieee)
// CHECK-PZ-F32-DYNAMIC: denormal_fpenv(positivezero, float: dynamic)
// CHECK-PZ-F32-PS: denormal_fpenv(positivezero, float: preservesign)
// CHECK-DYNAMIC-F32-PZ: denormal_fpenv(dynamic, float: positivezero)
// CHECK: CHECK-PS-F32-PZ: denormal_fpenv(preservesign, float: positivezero)

// CHECK-F32-IEEE: denormal_fpenv(float: ieee)
// CHECK-F32-PS: denormal_fpenv(float: preservesign)
// CHECK-F32-PZ: denormal_fpenv(float: positivezero)
// CHECK-F32-DYNAMIC: denormal_fpenv(float: dynamic)

int main(void) {
  return 0;
}
