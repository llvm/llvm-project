// Test without serialization:
// RUN: %clang_cc1 -w -ast-dump %s | FileCheck %s
//
// Test with serialization:
// RUN: %clang_cc1 -w -emit-pch -o %t %s
// RUN: %clang_cc1 -w -x c -include-pch %t -ast-dump-all /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck %s

// The cast construction code both for implicit and c-style casts is very
// different in C vs C++. This file is intended to test the C behavior.

// TODO: add tests covering the rest of the code in
// Sema::CheckAssignmentConstraints and Sema::PrepareScalarCast

// CHECK-LABEL: FunctionDecl {{.*}} cast_cvr_pointer
void cast_cvr_pointer(char volatile * __restrict * const * p) {
  char*** x;
  // CHECK: ImplicitCastExpr {{.*}} 'char ***' <NoOp>
  x = p;
  // CHECK: CStyleCastExpr {{.*}} 'char ***' <NoOp>
  x = (char***)p;
}

// CHECK-LABEL: FunctionDecl {{.*}} cast_pointer_type
void cast_pointer_type(char *p) {
  void *x;
  // CHECK: ImplicitCastExpr {{.*}} 'void *' <BitCast>
  x = p;
  // CHECK: CStyleCastExpr {{.*}} 'void *' <BitCast>
  x = (void*)p;
}
