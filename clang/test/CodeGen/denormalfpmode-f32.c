// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-NONE,CHECK-F32-NONE
// RUN: %clang_cc1 -fdenormal-fp-math=ieee %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-NONE,CHECK-F32-NONE
// RUN: %clang_cc1 -fdenormal-fp-math=preserve-sign %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-PS,CHECK-F32-NONE
// RUN: %clang_cc1 -fdenormal-fp-math=positive-zero %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-PZ,CHECK-F32-NONE
// RUN: %clang_cc1 -fdenormal-fp-math=dynamic %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-DYNAMIC,CHECK-F32-NONE

// RUN: %clang_cc1 -fdenormal-fp-math-f32=ieee %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-NONE,CHECK-F32-NONE
// RUN: %clang_cc1 -fdenormal-fp-math=ieee -fdenormal-fp-math-f32=ieee %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-NONE,CHECK-F32-NONE
// RUN: %clang_cc1 -fdenormal-fp-math=preserve-sign -fdenormal-fp-math-f32=ieee %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-PS,CHECK-F32-IEEE
// RUN: %clang_cc1 -fdenormal-fp-math=positive-zero -fdenormal-fp-math-f32=ieee %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-PZ,CHECK-F32-IEEE
// RUN: %clang_cc1 -fdenormal-fp-math=positive-zero -fdenormal-fp-math-f32=dynamic %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-PZ,CHECK-F32-DYNAMIC


// RUN: %clang_cc1 -fdenormal-fp-math-f32=preserve-sign %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-NONE,CHECK-F32-PS
// RUN: %clang_cc1 -fdenormal-fp-math=ieee -fdenormal-fp-math-f32=preserve-sign %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-NONE,CHECK-F32-PS
// RUN: %clang_cc1 -fdenormal-fp-math=preserve-sign -fdenormal-fp-math-f32=preserve-sign %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-PS,CHECK-F32-NONE
// RUN: %clang_cc1 -fdenormal-fp-math=positive-zero -fdenormal-fp-math-f32=preserve-sign %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-PZ,CHECK-F32-PS
// RUN: %clang_cc1 -fdenormal-fp-math=ieee -fdenormal-fp-math-f32=dynamic %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-NONE,CHECK-F32-DYNAMIC


// RUN: %clang_cc1 -fdenormal-fp-math-f32=positive-zero %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-NONE,CHECK-F32-PZ
// RUN: %clang_cc1 -fdenormal-fp-math-f32=dynamic %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-NONE,CHECK-F32-DYNAMIC
// RUN: %clang_cc1 -fdenormal-fp-math=ieee -fdenormal-fp-math-f32=positive-zero %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-NONE,CHECK-F32-PZ
// RUN: %clang_cc1 -fdenormal-fp-math=dynamic -fdenormal-fp-math-f32=positive-zero %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-DYNAMIC,CHECK-F32-PZ
// RUN: %clang_cc1 -fdenormal-fp-math=preserve-sign -fdenormal-fp-math-f32=positive-zero %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-PS,CHECK-F32-PZ
// RUN: %clang_cc1 -fdenormal-fp-math=positive-zero -fdenormal-fp-math-f32=positive-zero %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-PZ,CHECK-F32-NONE
// RUN: %clang_cc1 -fdenormal-fp-math=dynamic -fdenormal-fp-math-f32=dynamic %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-DYNAMIC,CHECK-F32-NONE


// CHECK-LABEL: main

// CHECK-ATTR: attributes #0 =
// CHECK-NONE-NOT:"denormal-fp-math"
// CHECK-IEEE: "denormal-fp-math"="ieee,ieee"
// CHECK-PS: "denormal-fp-math"="preserve-sign,preserve-sign"
// CHECK-PZ: "denormal-fp-math"="positive-zero,positive-zero"
// CHECK-DYNAMIC: "denormal-fp-math"="dynamic,dynamic"

// CHECK-F32-NONE-NOT:"denormal-fp-math-f32"
// CHECK-F32-IEEE: "denormal-fp-math-f32"="ieee,ieee"
// CHECK-F32-PS: "denormal-fp-math-f32"="preserve-sign,preserve-sign"
// CHECK-F32-PZ: "denormal-fp-math-f32"="positive-zero,positive-zero"


// CHECK-F32-DYNAMIC: "denormal-fp-math-f32"="dynamic,dynamic"

int main(void) {
  return 0;
}
