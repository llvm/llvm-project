

// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

typedef int * unspecified_ptr_t;
typedef int *__single single_ptr_t;
typedef int *__bidi_indexable bidi_ptr_t;
typedef int *__unsafe_indexable unsafe_ptr_t;
typedef int *__unsafe_indexable* unsafe_ptr_ptr_t;
typedef single_ptr_t *  single_ptr_ptr_t;
typedef unspecified_ptr_t *  unspecified_ptr_ptr_t;
typedef int *__null_terminated term_ptr_t;

unspecified_ptr_t g_single1;
single_ptr_t g_single2;
bidi_ptr_t g_bidi;
unsafe_ptr_t g_unsafe;
unspecified_ptr_ptr_t g_single_single;
term_ptr_t g_term;

void foo(unspecified_ptr_t a_single1,
         single_ptr_t a_single2,
         bidi_ptr_t a_bidi,
         unsafe_ptr_t a_unsafe,
         term_ptr_t a_term)
{
    unspecified_ptr_t l_bidi1;
    single_ptr_t l_single;
    bidi_ptr_t l_bidi2;
    unsafe_ptr_t l_unsafe;

    unspecified_ptr_t __single l_single2;
    unspecified_ptr_t __unsafe_indexable l_unsafe2;

    unsafe_ptr_ptr_t l_unsafe_bidi;
    single_ptr_ptr_t l_single_bidi1;
    unspecified_ptr_ptr_t l_single_bidi2;

    // initialize to avoid 'must be initialized error'
    term_ptr_t l_term = a_term;
}

// CHECK: |-TypedefDecl {{.*}} referenced unspecified_ptr_t 'int *'
// CHECK: | `-PointerType {{.*}} 'int *'
// CHECK: |   `-BuiltinType {{.*}} 'int'
// CHECK: |-TypedefDecl {{.*}} referenced single_ptr_t 'int *__single'
// CHECK: | `-PointerType {{.*}} 'int *__single'
// CHECK: |   `-BuiltinType {{.*}} 'int'
// CHECK: |-TypedefDecl {{.*}} referenced bidi_ptr_t 'int *__bidi_indexable'
// CHECK: | `-PointerType {{.*}} 'int *__bidi_indexable'
// CHECK: |   `-BuiltinType {{.*}} 'int'
// CHECK: |-TypedefDecl {{.*}} referenced unsafe_ptr_t 'int *__unsafe_indexable'
// CHECK: | `-PointerType {{.*}} 'int *__unsafe_indexable'
// CHECK: |   `-BuiltinType {{.*}} 'int'
// CHECK: |-TypedefDecl {{.*}} referenced unsafe_ptr_ptr_t 'int *__unsafe_indexable*'
// CHECK: | `-PointerType {{.*}} 'int *__unsafe_indexable*'
// CHECK: |   `-PointerType {{.*}} 'int *__unsafe_indexable'
// CHECK: |     `-BuiltinType {{.*}} 'int'
// CHECK: |-TypedefDecl {{.*}} referenced single_ptr_ptr_t 'single_ptr_t *'
// CHECK: | `-PointerType {{.*}} 'single_ptr_t *'
// CHECK: |   `-TypedefType {{.*}} 'single_ptr_t' sugar
// CHECK: |     |-Typedef {{.*}} 'single_ptr_t'
// CHECK: |     `-PointerType {{.*}} 'int *__single'
// CHECK: |       `-BuiltinType {{.*}} 'int'
// CHECK: |-TypedefDecl {{.*}} referenced unspecified_ptr_ptr_t 'unspecified_ptr_t *'
// CHECK: | `-PointerType {{.*}} 'unspecified_ptr_t *'
// CHECK: |   `-TypedefType {{.*}} 'unspecified_ptr_t' sugar
// CHECK: |     |-Typedef {{.*}} 'unspecified_ptr_t'
// CHECK: |     `-PointerType {{.*}} 'int *'
// CHECK: |       `-BuiltinType {{.*}} 'int'
// CHECK: |-TypedefDecl {{.*}} term_ptr_t 'int * __terminated_by(0)':'int *'
// CHECK: | `-ValueTerminatedType {{.*}} 'int * __terminated_by(0)' sugar
// CHECK: |   `-PointerType {{.*}} 'int *'
// CHECK: |     `-BuiltinType {{.*}} 'int'
// CHECK: |-VarDecl {{.*}} g_single1 'int *__single'
// CHECK: |-VarDecl {{.*}} g_single2 'single_ptr_t':'int *__single'
// CHECK: |-VarDecl {{.*}} g_bidi 'bidi_ptr_t':'int *__bidi_indexable'
// CHECK: |-VarDecl {{.*}} g_unsafe 'unsafe_ptr_t':'int *__unsafe_indexable'
// CHECK: |-VarDecl {{.*}} g_single_single 'int *__single*__single'
// CHECK: |-VarDecl {{.*}} g_term 'int *__single __terminated_by(0)':'int *__single'
// CHECK: `-FunctionDecl {{.*}} foo 'void (int *__single, single_ptr_t, bidi_ptr_t, unsafe_ptr_t, int *__single __terminated_by(0))'
// CHECK:   |-ParmVarDecl {{.*}} a_single1 'int *__single'
// CHECK:   |-ParmVarDecl {{.*}} a_single2 'single_ptr_t':'int *__single'
// CHECK:   |-ParmVarDecl {{.*}} a_bidi 'bidi_ptr_t':'int *__bidi_indexable'
// CHECK:   |-ParmVarDecl {{.*}} a_unsafe 'unsafe_ptr_t':'int *__unsafe_indexable'
// CHECK:   |-ParmVarDecl {{.*}} a_term 'int *__single __terminated_by(0)':'int *__single'
// CHECK:   `-CompoundStmt
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl {{.*}} l_bidi1 'int *__bidi_indexable'
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl {{.*}} l_single 'single_ptr_t':'int *__single'
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl {{.*}} l_bidi2 'bidi_ptr_t':'int *__bidi_indexable'
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl {{.*}} l_unsafe 'unsafe_ptr_t':'int *__unsafe_indexable'
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl {{.*}} l_single2 'unspecified_ptr_t __single':'int *__single'
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl {{.*}} l_unsafe2 'unspecified_ptr_t __unsafe_indexable':'int *__unsafe_indexable'
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl {{.*}} l_unsafe_bidi 'int *__unsafe_indexable*__bidi_indexable'
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl {{.*}} l_single_bidi1 'single_ptr_t *__bidi_indexable'
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl {{.*}} l_single_bidi2 'int *__single*__bidi_indexable'
// CHECK:     `-DeclStmt
// CHECK:       `-VarDecl {{.*}} l_term 'int *__single __terminated_by(0)':'int *__single'
