

// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fexperimental-bounds-safety-attributes -x c -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fexperimental-bounds-safety-attributes -x c++ -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fexperimental-bounds-safety-attributes -x objective-c -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fexperimental-bounds-safety-attributes -x objective-c++ -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

void Test() {
    int *__terminated_by(42) ptr = __unsafe_forge_terminated_by(int *, 17, 42);
}

// CHECK: FunctionDecl [[func_Test:0x[^ ]+]] {{.+}} Test
// CHECK:   `-CompoundStmt
// CHECK:     `-DeclStmt
// CHECK:       `-VarDecl [[var_ptr:0x[^ ]+]]
// CHECK:         `-ParenExpr
// CHECK:           `-CStyleCastExpr {{.+}} 'int *{{(__single)?}} __terminated_by(42)':'int *{{(__single)?}}' <BitCast>
// CHECK:             `-ForgePtrExpr {{.+}} 'void *{{(__single)?}} __terminated_by((42))':'void *{{(__single)?}}'
// CHECK:               |-ParenExpr
// CHECK:               | `-IntegerLiteral {{.+}} 17
// CHECK:               `-ParenExpr
// CHECK:                 `-IntegerLiteral {{.+}} 42

void Test2() {
    int **__terminated_by(0) ptr = __unsafe_forge_terminated_by(int **, 17, 0);
}

// CHECK: FunctionDecl [[func_Test2:0x[^ ]+]] {{.+}} Test2
// CHECK:   `-CompoundStmt
// CHECK:     `-DeclStmt
// CHECK:       `-VarDecl [[var_ptr_1:0x[^ ]+]]
// CHECK:         `-ParenExpr
// CHECK:           `-CStyleCastExpr {{.+}} 'int *{{(__single)?}}*{{(__single)?}} __terminated_by(0)':'int *{{(__single)?}}*{{(__single)?}}' <BitCast>
// CHECK:             `-ForgePtrExpr {{.+}} 'void *{{(__single)?}} __terminated_by((0))':'void *{{(__single)?}}'
// CHECK:               |-ParenExpr
// CHECK:               | `-IntegerLiteral {{.+}} 17
// CHECK:               `-ParenExpr
// CHECK:                 `-IntegerLiteral {{.+}} 0
