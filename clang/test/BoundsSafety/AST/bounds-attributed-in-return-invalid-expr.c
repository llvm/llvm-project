// FileCheck lines automatically generated using make-ast-dump-check-v2.py

// RUN: not %clang_cc1 -fbounds-safety -fbounds-safety-bringup-missing-checks=all -ast-dump %s 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -fbounds-safety -fno-bounds-safety-bringup-missing-checks=all -ast-dump %s 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -fbounds-safety -fbounds-safety-bringup-missing-checks=all -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -fbounds-safety -fno-bounds-safety-bringup-missing-checks=all -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

// Make sure we don't add bounds checks to RecoveryExprs
// CHECK-LABEL:`-FunctionDecl {{.+}} <{{.+}}, line:{{.+}}> line:{{.+}} no_bounds_check_expr_ret 'int *__single __sized_by(2)(int *__single __sized_by(3))'
// CHECK-NEXT:   |-ParmVarDecl {{.+}} p 'int *__single __sized_by(3)':'int *__single'
// CHECK-NEXT:   `-CompoundStmt {{.+}}
// CHECK-NEXT:     `-ReturnStmt {{.+}}
// CHECK-NEXT:       `-RecoveryExpr {{.+}} 'int *__single __sized_by(2)':'int *__single' contains-errors
// CHECK-NEXT:         `-FloatingLiteral {{.+}} 'float' 0.000000e+00
int* __sized_by(2) no_bounds_check_expr_ret(int* __sized_by(3) p) {
  return 0.0f;
}

