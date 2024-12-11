

// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s

// When processing the type for `b`, the types are processed from the beginning of the declaration,
// resulting in the type attribute being applied despite having been applied already

#include <ptrcheck.h>

typedef int * int_ptr_t;
void foo() {
  int_ptr_t __single a, b;
  int_ptr_t __bidi_indexable c, d;
  int_ptr_t __unsafe_indexable e, f;
  int_ptr_t __indexable g, h;
}

// CHECK: -FunctionDecl {{.*}} foo 'void ()'
// CHECK-NEXT:  `-CompoundStmt
// CHECK-NEXT:    |-DeclStmt
// CHECK-NEXT:    | |-VarDecl {{.*}} a 'int_ptr_t __single':'int *__single'
// CHECK-NEXT:    | `-VarDecl {{.*}} b 'int_ptr_t __single':'int *__single'
// CHECK-NEXT:    |-DeclStmt
// CHECK-NEXT:    | |-VarDecl {{.*}} c 'int_ptr_t __bidi_indexable':'int *__bidi_indexable'
// CHECK-NEXT:    | `-VarDecl {{.*}} d 'int_ptr_t __bidi_indexable':'int *__bidi_indexable'
// CHECK-NEXT:    |-DeclStmt
// CHECK-NEXT:    | |-VarDecl {{.*}} e 'int_ptr_t __unsafe_indexable':'int *__unsafe_indexable'
// CHECK-NEXT:    | `-VarDecl {{.*}} f 'int_ptr_t __unsafe_indexable':'int *__unsafe_indexable'
// CHECK-NEXT:    `-DeclStmt
// CHECK-NEXT:      |-VarDecl {{.*}} g 'int_ptr_t __indexable':'int *__indexable'
// CHECK-NEXT:      `-VarDecl {{.*}} h 'int_ptr_t __indexable':'int *__indexable'

