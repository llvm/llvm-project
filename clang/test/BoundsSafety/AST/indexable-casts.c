

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -fbounds-safety %s -o /dev/null
// RUN: %clang_cc1 -emit-llvm -fbounds-safety -O2 %s -o /dev/null

// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s -o /dev/null
// RUN: %clang_cc1 -emit-llvm -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -O2 %s -o /dev/null

#include <ptrcheck.h>

int foo() {
    int *__indexable p;
    p = (int *__indexable)(int *__bidi_indexable)p;
    char *__bidi_indexable cp = (char *__bidi_indexable)p;
    return 0;
}

// CHECK: |-DeclStmt
// CHECK: | `-VarDecl {{.*}} used p 'int *__indexable'
// CHECK: |-BinaryOperator {{.*}} 'int *__indexable' '='
// CHECK: | |-DeclRefExpr {{.*}} 'int *__indexable' lvalue Var {{.*}} 'p' 'int *__indexable'
// CHECK: | `-CStyleCastExpr {{.*}} 'int *__indexable' <BoundsSafetyPointerCast>
// CHECK: |   `-CStyleCastExpr {{.*}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK: |     `-ImplicitCastExpr {{.*}} 'int *__indexable' <LValueToRValue> part_of_explicit_cast
// CHECK: |       `-DeclRefExpr {{.*}} 'int *__indexable' lvalue Var {{.*}} 'p' 'int *__indexable'
// CHECK: |-DeclStmt
// CHECK: | `-VarDecl {{.*}} cp 'char *__bidi_indexable' cinit
// CHECK: |   `-CStyleCastExpr {{.*}} 'char *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK: |     `-ImplicitCastExpr {{.*}} 'char *__indexable' <BitCast> part_of_explicit_cast
// CHECK: |       `-ImplicitCastExpr {{.*}} 'int *__indexable' <LValueToRValue> part_of_explicit_cast
// CHECK: |         `-DeclRefExpr {{.*}} 'int *__indexable' lvalue Var {{.*}} 'p' 'int *__indexable'
