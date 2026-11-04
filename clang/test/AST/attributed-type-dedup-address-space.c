// RUN: %clang_cc1 -emit-pch -o %t.pch %s
// RUN: llvm-bcanalyzer --dump --disable-histogram %t.pch | FileCheck %s

__attribute__((address_space(1))) int *a;
__attribute__((address_space(1))) int *b;
__attribute__((address_space(2))) int *c;

// CHECK-COUNT-2: <TYPE_ATTRIBUTED
// CHECK-NOT:     <TYPE_ATTRIBUTED
