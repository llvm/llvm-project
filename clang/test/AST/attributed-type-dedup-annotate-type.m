// RUN: %clang_cc1 -emit-pch -o %t.pch %s
// RUN: llvm-bcanalyzer --dump --disable-histogram %t.pch | FileCheck %s

// Only two unique string arguments are present. Correspondingly,
// we should have two TYPE_ATTRIBUTED records.

int *[[clang::annotate_type("foo")]] a;
int *[[clang::annotate_type("foo")]] b;
int *[[clang::annotate_type("bar")]] c;

// CHECK-COUNT-2: <TYPE_ATTRIBUTED
// CHECK-NOT:     <TYPE_ATTRIBUTED
