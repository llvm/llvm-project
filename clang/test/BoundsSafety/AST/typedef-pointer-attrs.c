

// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>
#include "mock-typedef-header.h"

typedef int * giptr_t;
typedef int ** giptr_ptr_t;
typedef siptr_t *  gptr_siptr_t;

giptr_t gip;
giptr_ptr_t gipp;
gptr_siptr_t gsipp;

void foo() {
    giptr_t gip_local;
    giptr_ptr_t gipp_local;

    typedef char ** lcptr_ptr_t;
    lcptr_ptr_t lcptr_local;

    siptr_t sip_local;

    typedef int ** giptr_ptr_t;
    giptr_ptr_t gipp_local_redecl;
}

// CHECK: |-TypedefDecl {{.*}} referenced siptr_t 'int *'
// CHECK: | `-PointerType {{.*}} 'int *'
// CHECK: |   `-BuiltinType {{.*}} 'int'
// CHECK: |-TypedefDecl {{.*}} siptr_ptr_t 'int **'
// CHECK: | `-PointerType {{.*}} 'int **'
// CHECK: |   `-PointerType {{.*}} 'int *'
// CHECK: |     `-BuiltinType {{.*}} 'int'
// CHECK: |-TypedefDecl {{.*}} referenced giptr_t 'int *'
// CHECK: | `-PointerType {{.*}} 'int *'
// CHECK: |   `-BuiltinType {{.*}} 'int'
// CHECK: |-TypedefDecl {{.*}} referenced giptr_ptr_t 'int **'
// CHECK: | `-PointerType {{.*}} 'int **'
// CHECK: |   `-PointerType {{.*}} 'int *'
// CHECK: |     `-BuiltinType {{.*}} 'int'
// CHECK: |-TypedefDecl {{.*}} gptr_siptr_t 'siptr_t *'
// CHECK: | `-PointerType {{.*}} 'siptr_t *'
// CHECK: |   `-TypedefType {{.*}} 'siptr_t' sugar
// CHECK: |     |-Typedef {{.*}} 'siptr_t'
// CHECK: |     `-PointerType {{.*}} 'int *'
// CHECK: |       `-BuiltinType {{.*}} 'int'
// CHECK: |-VarDecl {{.*}} gip 'int *__single'
// CHECK: |-VarDecl {{.*}} gipp 'int *__single*__single'
// CHECK: |-VarDecl {{.*}} gsipp 'int *__single*__single'
// CHECK: `-FunctionDecl {{.*}} foo 'void ()'
// CHECK:   `-CompoundStmt
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl {{.*}} gip_local 'int *__bidi_indexable'
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl {{.*}} gipp_local 'int *__single*__bidi_indexable'
// CHECK:     |-DeclStmt
// CHECK:     | `-TypedefDecl {{.*}} referenced lcptr_ptr_t 'char **'
// CHECK:     |   `-PointerType {{.*}} 'char **'
// CHECK:     |     `-PointerType {{.*}} 'char *'
// CHECK:     |       `-BuiltinType {{.*}} 'char'
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl {{.*}} lcptr_local 'char *__single*__bidi_indexable'
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl {{.*}} sip_local 'int *__bidi_indexable'
// CHECK:     |-DeclStmt
// CHECK:     | `-TypedefDecl {{.*}} referenced giptr_ptr_t 'int **'
// CHECK:     |   `-PointerType {{.*}} 'int **'
// CHECK:     |     `-PointerType {{.*}} 'int *'
// CHECK:     |       `-BuiltinType {{.*}} 'int'
// CHECK:     `-DeclStmt
// CHECK:       `-VarDecl {{.*}} gipp_local_redecl 'int *__single*__bidi_indexable'
