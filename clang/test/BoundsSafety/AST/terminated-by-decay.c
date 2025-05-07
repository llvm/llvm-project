
// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

// __terminated_by arrays should decay to __terminated_by __single pointers.

// CHECK:      ImplicitCastExpr {{.+}} 'int *__single __terminated_by(0)':'int *__single' <ArrayToPointerDecay>
// CHECK-NEXT: `-DeclRefExpr {{.+}} 'int[__terminated_by(0) 3]':'int[3]' lvalue Var {{.+}} 'array' 'int[__terminated_by(0) 3]':'int[3]'
void null(void) {
  int array[__null_terminated 3] = {1, 2, 0};
  (void)array;
}

// CHECK:      ImplicitCastExpr {{.+}} 'int *__single __terminated_by(42)':'int *__single' <ArrayToPointerDecay>
// CHECK-NEXT: `-DeclRefExpr {{.+}} 'int[__terminated_by(42) 3]':'int[3]' lvalue Var {{.+}} 'array2' 'int[__terminated_by(42) 3]':'int[3]'
void _42(void) {
  int array2[__terminated_by(42) 3] = {1, 2, 42};
  (void)array2;
}

// CHECK:      ImplicitCastExpr {{.+}} 'const int *__single __terminated_by(0)':'const int *__single' <ArrayToPointerDecay>
// CHECK-NEXT: `-DeclRefExpr {{.+}} 'const int[__terminated_by(0) 3]':'const int[3]' lvalue Var {{.+}} 'array' 'const int[__terminated_by(0) 3]':'const int[3]'
void quals(void) {
  const int array[__null_terminated 3] = {1, 2, 0};
  (void)array;
}
