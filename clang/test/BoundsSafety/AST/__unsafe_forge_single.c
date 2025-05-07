

// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fexperimental-bounds-safety-attributes -x c -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fexperimental-bounds-safety-attributes -x c++ -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fexperimental-bounds-safety-attributes -x objective-c -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fexperimental-bounds-safety-attributes -x objective-c++ -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

// CHECK: FunctionDecl {{.+}} Test
// CHECK: `-CompoundStmt
// CHECK:   `-DeclStmt
// CHECK:     `-VarDecl {{.+}} ptr 'int *__single'
// CHECK:       `-ParenExpr {{.+}} 'int *__single'
// CHECK:         `-CStyleCastExpr {{.+}} 'int *__single'{{.*}} <BitCast>
// CHECK:           `-ForgePtrExpr {{.+}} 'void *__single'
// CHECK:             |-ParenExpr {{.+}} 'int'
// CHECK:             | `-IntegerLiteral {{.+}} 'int' 17
// CHECK:             |-<<<NULL>>>
// CHECK:             `-<<<NULL>>>
void Test(void) {
  int *__single ptr = __unsafe_forge_single(int *, 17);
}
